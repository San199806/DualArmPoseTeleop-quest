import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from dual_arm_pose_teleop.model_loader import _load_windows_unicode_xml


class ModelPathTests(unittest.TestCase):
    def test_unicode_include_and_duplicate_asset_basenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)/'模型'
            root.mkdir()
            mesh='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
            for folder in ('one','two'):
                (root/folder).mkdir()
                (root/folder/'shape.obj').write_text(mesh,encoding='utf-8')
            (root/'parts.xml').write_text('<mujoco><asset><mesh name="a" file="one/shape.obj"/>'
                '<mesh name="b" file="two/shape.obj" scale="2 2 2"/></asset></mujoco>',encoding='utf-8')
            path=root/'模型.xml'
            path.write_text('<mujoco><include file="parts.xml"/><worldbody><geom type="mesh" mesh="a"/>'
                '<geom type="mesh" mesh="b" pos="3 0 0"/></worldbody></mujoco>',encoding='utf-8')
            model=_load_windows_unicode_xml(path)
            self.assertEqual(model.nmesh,2)
            self.assertEqual(model.ngeom,2)
            self.assertGreater(model.geom_rbound[1],model.geom_rbound[0]*1.9)

    def test_include_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'cycle.xml'
            path.write_text('<mujoco><include file="cycle.xml"/></mujoco>')
            with self.assertRaisesRegex(ValueError,'Cyclic'):
                _load_windows_unicode_xml(path)
