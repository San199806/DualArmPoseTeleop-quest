import pathlib
import sys
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from dual_arm_pose_teleop.collision import CollisionGuard


class CollisionTests(unittest.TestCase):
    def fixture(self, obstacle=False, **settings):
        scene = '<geom name="obstacle" type="sphere" size=".05"/>' if obstacle else ''
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><default>
        <geom contype="0" conaffinity="0"/></default><worldbody>{scene}
        <body name="left" pos="-.3 0 0"><joint name="l" type="slide" axis="1 0 0"/>
        <geom name="lg" type="sphere" size=".05" mass="1"/></body>
        <body name="right" pos=".3 0 0"><joint name="r" type="slide" axis="1 0 0"/>
        <geom name="rg" type="sphere" size=".05" mass="1"/></body>
        </worldbody></mujoco>''')
        arms = {'left':{'joint_names':['l']},'right':{'joint_names':['r']}}
        guard = CollisionGuard(model,arms,{'enabled':True,**settings})
        return model,guard

    def test_zero_contact_masks_do_not_disable_detection(self):
        model,guard = self.fixture()
        self.assertTrue(guard.check(np.zeros(2)).safe)
        result = guard.check(np.array([.5,0.]))
        self.assertFalse(result.safe)
        self.assertIn('left/lg',result.message)
        self.assertTrue(np.all(model.geom_contype == 0))

    def test_safe_endpoints_but_colliding_path(self):
        _,guard = self.fixture(obstacle=True)
        start,end = np.array([0.,1.]),np.array([.6,1.])
        self.assertTrue(guard.check(start).safe)
        self.assertTrue(guard.check(end).safe)
        self.assertFalse(guard.check_path(start,end).safe)

    def test_invalid_pose_and_excessive_path_fail_closed(self):
        _,guard = self.fixture(max_path_samples=2)
        self.assertFalse(guard.check([np.nan,0]).safe)
        self.assertFalse(guard.check_path([0,0],[-.5,0]).safe)

    def test_clearance_not_just_penetration(self):
        _,guard = self.fixture()
        self.assertFalse(guard.check([.49,0]).safe)
        self.assertTrue(guard.check([.47,0]).safe)

    def test_constraint_sign_and_displacement_units(self):
        model,guard = self.fixture()
        data = mujoco.MjData(model)
        data.qpos[:] = [.46,0]
        mujoco.mj_forward(model,data)
        constraint = guard.make_limit().compute_qp_inequalities(SimpleNamespace(data=data),.005)
        np.testing.assert_allclose(constraint.G,[[1,-1]],atol=1e-8)
        np.testing.assert_allclose(constraint.h,[.01],atol=1e-8)
        # Moving together preserves clearance; moving left towards right is bounded.
        self.assertTrue(np.all(constraint.G@np.array([.01,.01]) <= constraint.h))
        self.assertTrue(np.any(constraint.G@np.array([.02,0]) > constraint.h))

    def test_disabled_is_explicit(self):
        _,guard = self.fixture(enabled=False)
        self.assertTrue(guard.check([.5,0]).safe)


if __name__ == '__main__':
    unittest.main()
