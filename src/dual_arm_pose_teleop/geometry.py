import numpy as np
from scipy.spatial.transform import Rotation


def normalize(vector, fallback=None):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length > 1e-8:
        return vector / length
    if fallback is None:
        raise ValueError("Cannot normalize a zero-length vector")
    return normalize(fallback)


def axis_vector(axis_name):
    sign = -1.0 if axis_name.startswith("-") else 1.0
    name = axis_name.lstrip("-")
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if name not in axes:
        raise ValueError(f"Unsupported axis: {axis_name}")
    return sign * axes[name]


def make_arm_frame(upper_direction, forearm_direction):
    """Create a stable frame whose X axis follows the upper arm."""
    x_axis = normalize(upper_direction)
    y_axis = np.asarray(forearm_direction, dtype=np.float64).copy()
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-6:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(reference, x_axis)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        y_axis = reference - np.dot(reference, x_axis) * x_axis
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def project_two_link(
    shoulder,
    raw_elbow,
    raw_wrist,
    upper_length,
    forearm_length,
    maximum_reach_ratio=0.97,
    margin=0.01,
    fallback_elbow_direction=None,
):
    """Project raw elbow/wrist positions onto an exactly feasible two-link arm."""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    raw_elbow = np.asarray(raw_elbow, dtype=np.float64)
    raw_wrist = np.asarray(raw_wrist, dtype=np.float64)
    wrist_ray = raw_wrist - shoulder
    if np.linalg.norm(wrist_ray) < 1e-8:
        wrist_ray = raw_elbow - shoulder
    wrist_direction = normalize(wrist_ray, [1.0, 0.0, 0.0])

    minimum_reach = abs(upper_length - forearm_length) + margin
    maximum_reach = min(
        (upper_length + forearm_length) * maximum_reach_ratio,
        upper_length + forearm_length - margin,
    )
    distance = np.clip(np.linalg.norm(wrist_ray), minimum_reach, maximum_reach)
    wrist = shoulder + distance * wrist_direction

    along = (
        upper_length ** 2 - forearm_length ** 2 + distance ** 2
    ) / (2.0 * distance)
    radius = np.sqrt(max(upper_length ** 2 - along ** 2, 0.0))
    center = shoulder + along * wrist_direction

    elbow_direction = raw_elbow - center
    elbow_direction -= np.dot(elbow_direction, wrist_direction) * wrist_direction
    if np.linalg.norm(elbow_direction) < 1e-6 and fallback_elbow_direction is not None:
        elbow_direction = np.asarray(fallback_elbow_direction, dtype=np.float64).copy()
        elbow_direction -= np.dot(elbow_direction, wrist_direction) * wrist_direction
    if np.linalg.norm(elbow_direction) < 1e-6:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(reference, wrist_direction)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        elbow_direction = np.cross(wrist_direction, reference)
    elbow_direction = normalize(elbow_direction)
    if fallback_elbow_direction is not None:
        previous = np.asarray(fallback_elbow_direction, dtype=np.float64).copy()
        previous -= np.dot(previous, wrist_direction) * wrist_direction
        if np.linalg.norm(previous) > 1e-6 and np.dot(elbow_direction, previous) < 0.0:
            elbow_direction = -elbow_direction
    elbow = center + radius * elbow_direction
    return elbow, wrist, elbow_direction
