"""Offline checks of the installed Baxter model, without SDK or Redis."""
import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dual_arm_pose_teleop.configuration import load_configuration
from dual_arm_pose_teleop.collision import CollisionGuard
from dual_arm_pose_teleop.ik_solver import DualArmIKSolver
from dual_arm_pose_teleop.model_loader import load_robot_model
from dual_arm_pose_teleop.types import Pose, ArmTarget, DualArmTargets


class BaxterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_configuration(ROOT/'config'/'teleop.json').values
        cls.model = load_robot_model(ROOT/cls.config['model_xml'])

    def test_only_baxter_installed(self):
        self.assertEqual([p.stem for p in (ROOT/'robots').glob('*.json')], ['baxter'])
        self.assertEqual([p.name for p in (ROOT/'assets').iterdir() if p.is_dir()], ['baxter'])
        self.assertEqual(self.model.nq, 15)
        self.assertEqual(sum(len(a['joint_names']) for a in self.config['arms'].values()), 14)

    def test_mjcf_matches_independent_urdf_fk_and_joint_limits(self):
        root = ET.parse(ROOT/'assets/baxter/source/baxter_description/urdf/baxter.urdf').getroot()
        joints = root.findall('joint')
        rng = np.random.default_rng(23)
        data = mujoco.MjData(self.model)
        for _ in range(8):
            values = {}
            for joint in joints:
                if joint.get('type') == 'fixed':
                    continue
                j = self.model.joint(joint.get('name')).id
                lo, hi = [float(joint.find('limit').get(k)) for k in ('lower','upper')]
                np.testing.assert_allclose(self.model.jnt_range[j], [lo,hi], atol=1e-10)
                values[joint.get('name')] = rng.uniform(lo,hi)
                data.qpos[self.model.jnt_qposadr[j]] = values[joint.get('name')]
            mujoco.mj_forward(self.model,data)
            transforms = {'base': np.eye(4)}
            pending = joints.copy()
            while pending:
                progress = False
                for joint in pending.copy():
                    parent = joint.find('parent').get('link')
                    if parent not in transforms:
                        continue
                    node = joint.find('origin')
                    matrix = np.eye(4)
                    if node is not None:
                        matrix[:3,:3] = Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
                        matrix[:3,3] = np.fromstring(node.get('xyz','0 0 0'),sep=' ')
                    if joint.get('type') != 'fixed':
                        axis = np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                        matrix[:3,:3] = matrix[:3,:3] @ Rotation.from_rotvec(axis*values[joint.get('name')]).as_matrix()
                    child = joint.find('child').get('link')
                    transforms[child] = transforms[parent] @ matrix
                    pending.remove(joint)
                    progress = True
                self.assertTrue(progress)
            for name, matrix in transforms.items():
                b = self.model.body(name).id
                np.testing.assert_allclose(data.xpos[b],matrix[:3,3],atol=1e-8)
                np.testing.assert_allclose(data.xmat[b].reshape(3,3),matrix[:3,:3],atol=1e-8)

    def test_initial_pose_is_symmetric_and_safe(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.config['initial_qpos']
        mujoco.mj_forward(self.model,data)
        for key in ('shoulder','elbow','wrist'):
            a,b = [data.xpos[self.model.body(self.config['frames'][side+'_'+key]).id]
                   for side in ('left','right')]
            np.testing.assert_allclose(a*np.array([1,-1,1]),b,atol=.005)
        guard = CollisionGuard(self.model,self.config['arms'],self.config['ik']['collision'])
        report = guard.check(data.qpos)
        self.assertTrue(report.safe, report.message)
        self.assertTrue(all(self.model.geom_group[g] == 3 for pair in guard.pairs for g in pair))
        for j in range(self.model.njnt):
            self.assertTrue(self.model.jnt_range[j,0] <= data.qpos[self.model.jnt_qposadr[j]] <= self.model.jnt_range[j,1])

    def test_arm_command_order_and_small_ik_motion(self):
        c = self.config
        ik = DualArmIKSolver(ROOT/c['model_xml'],c['frames'],c['arms'],c['ik'],c['initial_qpos'],c['runtime']['command_output'])
        poses = ik.calibration_poses()
        arms = []
        for side in ('left','right'):
            wrist = poses[side+'_wrist']
            arms.append(ArmTarget(poses[side+'_elbow'],Pose(wrist.position+np.array([.003,0,0]),wrist.rotation)))
        targets = DualArmTargets(*arms)
        result = ik.solve(targets)
        self.assertIsNone(ik.collision_blocked)
        self.assertEqual(ik.command_positions(result).shape,(14,))
        np.testing.assert_array_equal(ik.command_positions(np.arange(15)),np.r_[np.arange(8,15),np.arange(1,8)])
        self.assertAlmostEqual(result[self.model.jnt_qposadr[self.model.joint('head_pan').id]],0)
        for side in ('left','right'):
            self.assertLess(np.linalg.norm(ik.body_pose(c['frames'][side+'_wrist']).position-getattr(targets,side).wrist.position),.01)

    def test_model_exclusions_keep_cross_arm_and_distal_base_checks(self):
        c = self.config
        guard = CollisionGuard(self.model,c['arms'],c['ik']['collision'])
        pairs = {frozenset((self.model.body(int(self.model.geom_bodyid[a])).name,
                            self.model.body(int(self.model.geom_bodyid[b])).name))
                 for a,b in guard.pairs}
        self.assertNotIn(frozenset(('torso','left_upper_shoulder')),pairs)
        for side in ('left','right'):
            for part in ('hand','wrist','upper_forearm_visual'):
                self.assertIn(frozenset(('torso',side+'_'+part)),pairs)
                self.assertIn(frozenset(('pedestal',side+'_'+part)),pairs)
        self.assertIn(frozenset(('left_hand','right_hand')),pairs)
        self.assertIn(frozenset(('left_upper_elbow','right_upper_elbow')),pairs)
        unsafe = np.array([0,.6,.7,0,1.5,0,-1,0,-.6,.7,0,1.5,0,-1,0])
        self.assertFalse(guard.check(unsafe).safe)

    def test_invalid_or_empty_collision_group_fails_closed(self):
        for groups in ([],[6],[True],[5]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                CollisionGuard(self.model,self.config['arms'],dict(self.config['ik']['collision'],geom_groups=groups))


if __name__ == '__main__':
    unittest.main()
