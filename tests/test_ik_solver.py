import pathlib
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.ik_solver import DualArmIKSolver
from dual_arm_pose_teleop.types import Pose, ArmTarget, DualArmTargets


class IKSolverTests(unittest.TestCase):
    def test_qp_locks_free_base_with_nq_different_from_nv(self):
        xml = '''<mujoco><worldbody><body name="base"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body name="left" pos="0 .3 0"><joint name="L" axis="0 0 1" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 .4 0 0" size=".02" mass="1"/>
        <body name="left_tip" pos=".4 0 0"/></body>
        <body name="right" pos="0 -.3 0"><joint name="R" axis="0 0 1" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 .4 0 0" size=".02" mass="1"/>
        <body name="right_tip" pos=".4 0 0"/></body>
        </body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/"model.xml"
            path.write_text(xml)
            frames = {"base": "base", "left_shoulder": "left", "right_shoulder": "right",
                      "left_elbow": "left", "right_elbow": "right",
                      "left_wrist": "left_tip", "right_wrist": "right_tip"}
            config = {"solver":"daqp", "damping": .1, "iterations": 10,
                      "base_position_cost":100, "base_orientation_cost":100,
                      "elbow_position_cost":1, "wrist_position_cost":30, "wrist_orientation_cost":0}
            initial = np.array([0,0,0,1,0,0,0,0,0], dtype=float)
            solver = DualArmIKSolver(path, frames, {"left":{"joint_names":["L"]}, "right":{"joint_names":["R"]}}, config, initial)
            targets = {}
            for side in ("left","right"):
                wrist = solver.body_pose(frames[side+"_wrist"])
                targets[side] = ArmTarget(solver.body_pose(frames[side+"_elbow"]), Pose(wrist.position+np.array([0,.1,.1]), wrist.rotation))
            q = solver.solve(DualArmTargets(**targets))
            np.testing.assert_allclose(q[:7], initial[:7], atol=1e-9)
            self.assertGreater(np.linalg.norm(q[7:]), 1e-5)
            solver.configuration.update(q=initial)
            solver.max_solve_time_s = 1e-9
            solver.solve(DualArmTargets(**targets))
            self.assertEqual(solver.last_solve_iterations, 1)
            lock = solver.limits[-1]
            constraint = lock.compute_qp_inequalities(solver.configuration,.01)
            self.assertEqual(constraint.G.shape, (12,8))
