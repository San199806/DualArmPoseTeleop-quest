"""Robot-profile-driven pose teleoperation for dual-arm robots."""

__all__ = ["Pose", "SensorFrame", "PoseArmMapper"]


def __getattr__(name):
    # Configuration inspection must not require the simulation dependencies.
    if name == "PoseArmMapper":
        from .mapper import PoseArmMapper
        return PoseArmMapper
    if name in ("Pose", "SensorFrame"):
        from .types import Pose, SensorFrame
        return {"Pose": Pose, "SensorFrame": SensorFrame}[name]
    raise AttributeError(name)
