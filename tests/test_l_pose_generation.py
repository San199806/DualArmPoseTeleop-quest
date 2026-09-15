import pathlib
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
try:
    from dual_arm_pose_teleop.pose_symmetry import generate_l_pose
except ModuleNotFoundError:
    from pose_symmetry import generate_l_pose


class LPoseTests(unittest.TestCase):
    def fixture(self, lower=-1):
        arms = ""
        for side,y in (("left",.3),("right",-.3)):
            arms += f'''<body name="{side}s" pos="0 {y} 0">
            <joint name="{side}1" axis="0 1 0" range="{lower} 1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".02"/>
            <body name="{side}e" pos="0 0 -.3">
            <joint name="{side}2" axis="0 1 0" range="-2 2"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".02"/>
            <body name="{side}w" pos="0 0 -.3"/></body></body>'''
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
        <worldbody><body name="base">{arms}</body></worldbody></mujoco>''')
        config = {"initial_qpos":[max(lower,.2),-1.7,max(lower,.2),-1.7],
                  "frames":{"base":"base"},"arms":{}}
        for side in ("left","right"):
            config["arms"][side] = {"joint_names":[side+"1",side+"2"]}
            for key,suffix in (("shoulder","s"),("elbow","e"),("wrist","w")):
                config["frames"][side+"_"+key] = side+suffix
        return model,config,np.array(config["initial_qpos"])

    def test_strict(self):
        model,config,q = self.fixture()
        result = generate_l_pose(model,q,config,"+x")
        self.assertTrue(result.success,result.message)
        self.assertIn("Strict L pose",result.message)

    def test_approximate_and_repeatable(self):
        model,config,q = self.fixture(.1)
        result = generate_l_pose(model,q,config,"+x")
        self.assertTrue(result.success,result.message)
        self.assertIn("Approximate L pose",result.message)
        self.assertGreaterEqual(result.qpos[0],.1+np.deg2rad(2)-1e-8)
        self.assertGreaterEqual(result.qpos[2],.1+np.deg2rad(2)-1e-8)
        second = generate_l_pose(model,result.qpos,config,"+x")
        np.testing.assert_allclose(result.qpos,second.qpos,atol=1e-9)
        np.testing.assert_array_equal(q,config["initial_qpos"])

    def test_excessive_tilt_rejected(self):
        model,config,q = self.fixture(.7)
        result = generate_l_pose(model,q,config,"+x")
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.qpos,q)

    def test_sideways_axis_rejected(self):
        model,config,q = self.fixture()
        result = generate_l_pose(model,q,config,"+y")
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.qpos,q)

    def test_collision_rejection_returns_original(self):
        from dual_arm_pose_teleop.collision import CollisionReport
        model,config,q = self.fixture()
        result = generate_l_pose(model,q,config,"+x",
                                 lambda _: CollisionReport(False,-.01,'test','test collision'))
        self.assertFalse(result.success)
        self.assertIn('Collision rejected',result.message)
        np.testing.assert_array_equal(result.qpos,q)


if __name__ == "__main__":
    unittest.main()
