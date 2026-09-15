import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dual_arm_pose_teleop.configuration import load_configuration,save_initial_qpos


class ConfigurationTests(unittest.TestCase):
    def fixture(self,directory):
        root=pathlib.Path(directory)
        shutil.copytree(ROOT/'config',root/'config')
        shutil.copytree(ROOT/'robots',root/'robots')
        return root,root/'config'/'teleop.json'

    def test_daily_config_is_small_and_simulation_default(self):
        path=ROOT/'config'/'teleop.json'
        self.assertEqual(len(json.loads(path.read_text())),8)
        result=load_configuration(path)
        self.assertTrue(result.values['runtime']['simulation'])
        self.assertEqual(result.profile_path,ROOT/'robots'/'baxter.json')
        self.assertEqual(result.values['ik']['collision']['geom_groups'],[3])

    def test_robot_override_changes_profile_and_calibration_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.fixture(directory)
            other=json.loads((root/'robots'/'baxter.json').read_text())
            other['robot']='test_robot'
            (root/'robots'/'test_robot.json').write_text(json.dumps(other))
            result=load_configuration(path,robot='test_robot')
            self.assertEqual(result.values['robot'],'test_robot')
            self.assertIn('test_robot',result.values['runtime']['calibration_file'])

    def test_legacy_alias_and_full_profile(self):
        alias=load_configuration(ROOT/'robots'/'baxter.json')
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'old.json'
            old=alias.values.copy()
            path.write_text(json.dumps(old))
            loaded=load_configuration(path)
            self.assertEqual(loaded.values,old)
            self.assertEqual(loaded.profile_path,path)

    def test_daily_and_advanced_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.fixture(directory)
            raw=json.loads(path.read_text())
            raw['max_joint_velocity_rad_s']=.8
            raw['advanced']={'runtime':{'max_joint_velocity_rad_s':.6},'ik':{'iterations':4}}
            path.write_text(json.dumps(raw))
            loaded=load_configuration(path)
            self.assertEqual(loaded.values['runtime']['max_joint_velocity_rad_s'],.6)
            self.assertEqual(loaded.values['ik']['iterations'],4)
            self.assertEqual(loaded.values['ik']['collision']['geom_groups'],[3])

    def test_typo_and_bad_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _,path=self.fixture(directory)
            original=json.loads(path.read_text())
            for key,value in [('collison_enabled',False),('simulation','false'),
                              ('human_upper_arm_length_m',-1),('robot','../baxter')]:
                raw={**original,key:value};path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):load_configuration(path)

    def test_save_only_updates_robot_initial_qpos(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.fixture(directory)
            loaded=load_configuration(path)
            daily_bytes=path.read_bytes();devices=(root/'config'/'devices.json').read_bytes()
            original=json.loads(loaded.profile_path.read_text())
            q=loaded.values['initial_qpos'].copy();q[4]+=.01
            save_initial_qpos(loaded,q)
            saved=json.loads(loaded.profile_path.read_text())
            self.assertEqual(saved,{**original,'initial_qpos':q})
            self.assertEqual(path.read_bytes(),daily_bytes)
            self.assertEqual((root/'config'/'devices.json').read_bytes(),devices)
            self.assertNotIn('diagnostics',saved)

    def test_save_rejects_changed_robot_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            _,path=self.fixture(directory);loaded=load_configuration(path)
            raw=json.loads(loaded.profile_path.read_text());raw['model_xml']='different.xml'
            loaded.profile_path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):save_initial_qpos(loaded,loaded.values['initial_qpos'])
            self.assertEqual(json.loads(loaded.profile_path.read_text()),raw)

    def test_alias_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'cycle.json';path.write_text('{"robot_profile":"cycle.json"}')
            with self.assertRaises(ValueError):load_configuration(path)

    def test_cli_effective_config_overrides_without_devices(self):
        output=subprocess.check_output([sys.executable,'-B',str(ROOT/'run.py'),
            '--robot','baxter','--real','--redis-host','example.local','--print-config'],encoding='utf-8')
        result=json.loads(output)['effective']
        self.assertEqual(result['robot'],'baxter')
        self.assertFalse(result['runtime']['simulation'])
        self.assertEqual(result['runtime']['redis_host'],'example.local')


if __name__=='__main__':unittest.main()
