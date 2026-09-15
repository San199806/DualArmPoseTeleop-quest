import pathlib
import sys
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.pose_symmetry import symmetrize_pose, mirror_orientation, mirror_position


class SymmetryTests(unittest.TestCase):
    def model(self, tight=False):
        right_range = "-.001 .001" if tight else "-2 2"
        xml = f'''<mujoco><compiler angle="radian"/>
        <worldbody><body name="base"><freejoint/><geom type="sphere" size=".05" mass="1"/>
        <body name="ls" pos="0 .3 0"><joint name="l1" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
        <body name="le" pos=".3 0 0"><joint name="l2" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
        <body name="lw" pos=".3 0 0"/></body></body>
        <body name="rs" pos="0 -.3 0"><joint name="r1" axis="0 0 1" range="{right_range}"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
        <body name="re" pos=".3 0 0"><joint name="r2" axis="0 0 1" range="{right_range}"/>
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
        <body name="rw" pos=".3 0 0"/></body></body></body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        config = {"arms":{"left":{"joint_names":["l1","l2"]}, "right":{"joint_names":["r1","r2"]}},
                  "frames":{"left_shoulder":"ls","left_elbow":"le","left_wrist":"lw",
                            "right_shoulder":"rs","right_elbow":"re","right_wrist":"rw"}}
        q = model.qpos0.copy()
        q[:3] = [1,2,3]
        q[3:7] = Rotation.from_euler("xyz",[.2,.3,.4]).as_quat(scalar_first=True)
        q[7:] = [.4,-.8,0,0]
        return model, config, q

    def test_mirror_preserves_source_and_free_base(self):
        model, config, q = self.model()
        result = symmetrize_pose(model,q,config,"left")
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.qpos[:9], q[:9])
        np.testing.assert_allclose(result.qpos[9:],[-.4,.8],atol=1e-5)
        self.assertLess(max(result.position_errors_m.values()),1e-5)
        # Pure solver never mutates caller input.
        np.testing.assert_array_equal(q[9:],[0,0])

    def test_reverse_direction_preserves_reference(self):
        model, config, q = self.model()
        q[7:] = [0,0,-.4,.8]
        result = symmetrize_pose(model,q,config,"right")
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.qpos[:7],q[:7])
        np.testing.assert_array_equal(result.qpos[9:],q[9:])
        np.testing.assert_allclose(result.qpos[7:9],[.4,-.8],atol=1e-5)

    def test_unreachable_does_not_return_partial_pose(self):
        model, config, q = self.model(tight=True)
        result = symmetrize_pose(model,q,config,"left")
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.qpos,q)

    def test_orientation_remains_proper_with_different_mount_frames(self):
        a = Rotation.from_euler("x",.5).as_matrix()
        b = Rotation.from_euler("y",-.7).as_matrix()
        n = np.array([0.,1.,0.])
        np.testing.assert_allclose(mirror_orientation(a,a,b,n),b,atol=1e-12)
        r = mirror_orientation(Rotation.from_euler("z",.3).as_matrix()@a,a,b,n)
        self.assertAlmostEqual(np.linalg.det(r),1.0)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)

    def test_plane_reflection_is_involution(self):
        p = np.array([1.,2.,3.]); c=np.array([.1,.2,.3]); n=np.array([0.,1.,0.])
        np.testing.assert_allclose(mirror_position(mirror_position(p,c,n),c,n),p)
