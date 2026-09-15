from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: Rotation


@dataclass(frozen=True)
class SensorFrame:
    head: Pose
    waist: Pose
    left_elbow: Pose
    right_elbow: Pose
    left_controller: Pose
    right_controller: Pose


@dataclass(frozen=True)
class ArmTarget:
    elbow: Pose
    wrist: Pose


@dataclass(frozen=True)
class DualArmTargets:
    left: ArmTarget
    right: ArmTarget
