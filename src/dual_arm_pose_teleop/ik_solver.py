import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .types import Pose
from .collision import CollisionGuard
from .model_loader import load_robot_model
from mink.limits.limit import Constraint, Limit


class LockedDofs(Limit):
    def __init__(self, nv, indices):
        projection = np.eye(nv)[indices]
        self.G = np.vstack((projection, -projection))
        self.h = np.zeros(2 * len(indices))

    def compute_qp_inequalities(self, configuration, dt):
        return Constraint(G=self.G, h=self.h)


class DualArmIKSolver:
    """Robot-independent Mink wrapper for a pair of serial arms."""

    def __init__(
        self,
        model_xml,
        frames,
        arms,
        ik_config,
        initial_qpos,
        command_output="full_qpos",
    ):
        self.model = load_robot_model(model_xml,ik_config.get('collision'))
        self.configuration = mink.Configuration(self.model)
        qpos = np.asarray(initial_qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"initial_qpos has {qpos.size} values; model requires {self.model.nq}")
        self.initial_qpos = qpos.copy()
        self.configuration.update(q=qpos)
        self.frames = frames
        self.arm_qpos_indices = {
            side: self._resolve_arm_qpos_indices(arms[side]["joint_names"], side)
            for side in ("left", "right")
        }
        all_indices = self.arm_qpos_indices["left"] + self.arm_qpos_indices["right"]
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Left and right arm joint lists overlap")
        self.arm_dof_count = len(all_indices)
        self.command_output = command_output
        if command_output not in ("full_qpos", "arm_joints"):
            raise ValueError(f"Unsupported command_output: {command_output}")
        self.solver = ik_config["solver"]
        self.damping = float(ik_config["damping"])
        self.iterations = int(ik_config["iterations"])
        self.solution_tolerance = float(ik_config.get("solution_tolerance", 1e-4))
        self.lock_non_arm_qpos = bool(ik_config.get("lock_non_arm_qpos", True))
        arm_index_set = set(all_indices)
        self.fixed_qpos_indices = np.asarray(
            [index for index in range(self.model.nq) if index not in arm_index_set],
            dtype=int,
        )
        self.tasks = {
            "base": mink.FrameTask(
                frames["base"], "body",
                position_cost=ik_config["base_position_cost"],
                orientation_cost=ik_config["base_orientation_cost"],
                lm_damping=1.0,
            ),
            "left_elbow": mink.FrameTask(
                frames["left_elbow"], "body",
                position_cost=ik_config["elbow_position_cost"],
                orientation_cost=0.0,
                lm_damping=1.0,
            ),
            "left_wrist": mink.FrameTask(
                frames["left_wrist"], "body",
                position_cost=ik_config["wrist_position_cost"],
                orientation_cost=ik_config["wrist_orientation_cost"],
                lm_damping=1.0,
            ),
            "right_elbow": mink.FrameTask(
                frames["right_elbow"], "body",
                position_cost=ik_config["elbow_position_cost"],
                orientation_cost=0.0,
                lm_damping=1.0,
            ),
            "right_wrist": mink.FrameTask(
                frames["right_wrist"], "body",
                position_cost=ik_config["wrist_position_cost"],
                orientation_cost=ik_config["wrist_orientation_cost"],
                lm_damping=1.0,
            ),
        }
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.collision = CollisionGuard(self.model, arms, ik_config.get("collision"))
        self.collision_blocked = None
        if self.collision.enabled:
            self.limits.append(self.collision.make_limit())
            print(f"[Collision] enabled: {len(self.collision.pairs)} pairs, clearance={self.collision.minimum*1000:.1f} mm")
        arm_dofs = {
            int(self.model.jnt_dofadr[self.model.joint(name).id])
            for side in ("left", "right") for name in arms[side]["joint_names"]
        }
        if self.lock_non_arm_qpos:
            fixed = [i for i in range(self.model.nv) if i not in arm_dofs]
            if fixed:
                self.limits.append(LockedDofs(self.model.nv, fixed))
        self.tasks["base"].set_target(self._as_se3(self.body_pose(frames["base"])))

    def _resolve_arm_qpos_indices(self, joint_names, side):
        indices = []
        for name in joint_names:
            try:
                joint_id = self.model.joint(name).id
            except KeyError as error:
                raise ValueError(f"Unknown {side} arm joint: {name}") from error
            joint_type = self.model.jnt_type[joint_id]
            scalar_types = (
                mujoco.mjtJoint.mjJNT_SLIDE,
                mujoco.mjtJoint.mjJNT_HINGE,
            )
            if joint_type not in scalar_types:
                raise ValueError(f"Arm joint {name} is not a scalar hinge/slide joint")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        if not indices:
            raise ValueError(f"No {side} arm joints configured")
        return indices

    @staticmethod
    def _as_se3(pose):
        quat_wxyz = pose.rotation.as_quat(scalar_first=True)
        return mink.SE3.from_rotation_and_translation(
            mink.SO3(quat_wxyz), pose.position
        )

    def body_pose(self, body_name):
        body_id = self.model.body(body_name).id
        data = self.configuration.data
        return Pose(
            data.xpos[body_id].copy(),
            Rotation.from_matrix(data.xmat[body_id].reshape(3, 3).copy()),
        )

    def calibration_poses(self):
        return {
            "base": self.body_pose(self.frames["base"]),
            "left_shoulder": self.body_pose(self.frames["left_shoulder"]),
            "left_elbow": self.body_pose(self.frames["left_elbow"]),
            "left_wrist": self.body_pose(self.frames["left_wrist"]),
            "right_shoulder": self.body_pose(self.frames["right_shoulder"]),
            "right_elbow": self.body_pose(self.frames["right_elbow"]),
            "right_wrist": self.body_pose(self.frames["right_wrist"]),
        }

    def inject_real_arm_joints(self, arm_joints):
        """Use measured arm joints, in configured left-then-right order, for IK."""
        arm_joints = np.asarray(arm_joints, dtype=np.float64)
        if arm_joints.shape != (self.arm_dof_count,):
            raise ValueError(
                f"real_robot_joint_state has {arm_joints.size} values; "
                f"expected {self.arm_dof_count}"
            )
        if not np.all(np.isfinite(arm_joints)):
            raise ValueError("real_robot_joint_state contains NaN or infinity")

        qpos = self.configuration.data.qpos.copy()
        cursor = 0
        for side in ("left", "right"):
            indices = self.arm_qpos_indices[side]
            qpos[indices] = arm_joints[cursor : cursor + len(indices)]
            cursor += len(indices)
        self.configuration.update(q=qpos)

    def reset_to_initial(self):
        self.configuration.update(q=self.initial_qpos.copy())

    def command_positions(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if self.command_output == "full_qpos":
            return qpos.copy()
        indices = self.arm_qpos_indices["left"] + self.arm_qpos_indices["right"]
        return qpos[indices].copy()

    def expand_command(self, command, reference):
        if self.command_output == "full_qpos":
            return command.copy()
        q = reference.copy()
        q[self.arm_qpos_indices["left"] + self.arm_qpos_indices["right"]] = command
        return q

    def command_bounds(self):
        lower = self.initial_qpos.copy()
        upper = self.initial_qpos.copy()
        for indices in self.arm_qpos_indices.values():
            for index in indices:
                joint = int(np.flatnonzero(self.model.jnt_qposadr == index)[0])
                if self.model.jnt_limited[joint]:
                    lower[index], upper[index] = self.model.jnt_range[joint]
                else:
                    lower[index], upper[index] = -np.inf, np.inf
        return self.command_positions(lower), self.command_positions(upper)

    def solve(self, targets):
        self.collision_blocked = None
        start = self.configuration.data.qpos.copy()
        report = self.collision.check(start)
        if not report.safe:
            self.collision_blocked = report.message
            return start
        identity = Rotation.identity()
        self.tasks["left_elbow"].set_target(
            self._as_se3(Pose(targets.left.elbow.position, identity))
        )
        self.tasks["left_wrist"].set_target(self._as_se3(targets.left.wrist))
        self.tasks["right_elbow"].set_target(
            self._as_se3(Pose(targets.right.elbow.position, identity))
        )
        self.tasks["right_wrist"].set_target(self._as_se3(targets.right.wrist))
        task_list = list(self.tasks.values())
        dt = self.model.opt.timestep
        for _ in range(self.iterations):
            previous = self.configuration.data.qpos.copy()
            try:
                velocity = mink.solve_ik(
                    self.configuration, task_list, dt, self.solver,
                    self.damping, self.limits,
                )
            except Exception as error:
                if not self.collision.enabled:
                    raise
                self.configuration.update(q=previous)
                self.collision_blocked = f"Constrained IK failed: {error}"
                break
            self.configuration.integrate_inplace(velocity, dt)
            if self.lock_non_arm_qpos and self.fixed_qpos_indices.size:
                qpos = self.configuration.data.qpos.copy()
                qpos[self.fixed_qpos_indices] = self.initial_qpos[
                    self.fixed_qpos_indices
                ]
                self.configuration.update(q=qpos)
            if self.collision.enabled:
                report = self.collision.check_path(previous,self.configuration.data.qpos)
                if not report.safe:
                    self.configuration.update(q=previous)
                    self.collision_blocked = report.message
                    break
            if np.linalg.norm(velocity) * dt < self.solution_tolerance:
                break
        return self.configuration.data.qpos.copy()
