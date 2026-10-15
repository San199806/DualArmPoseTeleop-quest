"""Generate a conservative robot-profile candidate from a MuJoCo model."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import mujoco
import numpy as np

from .model_loader import load_robot_model


SIDES = ("left", "right")
# Compare as ints. NumPy joint types and MuJoCo enums compare with ==, but
# `in` rejects the NumPy bool that equality returns.
SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


@dataclass
class GeneratedProfile:
    values: dict
    warnings: list[str]
    report: dict


def _side(name: str) -> str | None:
    value = (name or "").lower()
    for side, short in (("left", "l"), ("right", "r")):
        if (re.search(rf"(^|[^a-z0-9]){side}([^a-z0-9]|$)", value)
                or re.search(rf"^{short}[_\-.]", value)):
            return side
    return None


def _ancestors(model, body_id):
    result = []
    while body_id:
        result.append(int(body_id))
        body_id = int(model.body_parentid[body_id])
    result.append(0)
    return result


def _depths(model):
    return [len(_ancestors(model, body)) for body in range(model.nbody)]


def _joint_name(model, joint_id):
    name = model.joint(joint_id).name
    if not name:
        raise ValueError(f"All arm joints must be named; joint id {joint_id} is unnamed")
    return name


def _resolve_joints(model, names, side):
    if names:
        result = []
        for name in names:
            try:
                result.append(int(model.joint(name).id))
            except KeyError as error:
                raise ValueError(f"Unknown {side} arm joint: {name}") from error
    else:
        candidates = []
        for joint in range(model.njnt):
            if int(model.jnt_type[joint]) not in SCALAR_JOINTS:
                continue
            body = int(model.jnt_bodyid[joint])
            if _side(_joint_name(model, joint)) == side or _side(model.body(body).name or "") == side:
                candidates.append(joint)
        if not candidates:
            raise ValueError(f"Could not infer {side} arm joints from names")
        depths = _depths(model)
        chains = []
        for terminal in candidates:
            ancestors = set(_ancestors(model, int(model.jnt_bodyid[terminal])))
            chain = [joint for joint in candidates if int(model.jnt_bodyid[joint]) in ancestors]
            chain.sort(key=lambda joint: (depths[int(model.jnt_bodyid[joint])], int(model.jnt_qposadr[joint])))
            chains.append(chain)
        maximum = max(map(len, chains))
        unique = {tuple(chain) for chain in chains if len(chain) == maximum}
        if len(unique) != 1:
            options = [[_joint_name(model, joint) for joint in chain] for chain in sorted(unique)]
            raise ValueError(f"Ambiguous {side} arm chains: {options}; use --{side}-joints")
        result = list(unique.pop())
    if len(result) < 3:
        raise ValueError(f"{side} arm needs at least three scalar joints")
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate joint in {side} arm")
    if any(int(model.jnt_type[joint]) not in SCALAR_JOINTS for joint in result):
        raise ValueError(f"{side} arm contains a non-scalar joint")
    depths = _depths(model)
    if not names:
        result.sort(key=lambda joint: (depths[int(model.jnt_bodyid[joint])], int(model.jnt_qposadr[joint])))
    previous = None
    for joint in result:
        body = int(model.jnt_bodyid[joint])
        if previous is not None and previous not in _ancestors(model, body):
            raise ValueError(f"{side} joints do not form one serial kinematic chain")
        previous = body
    return result


def _descendants(model, root):
    return [body for body in range(1, model.nbody) if root in _ancestors(model, body)]


def _semantic_body(model, side, semantic, joints, explicit, warnings):
    if explicit:
        try:
            return model.body(explicit).name
        except KeyError as error:
            raise ValueError(f"Unknown {side}_{semantic} body: {explicit}") from error
    root = int(model.jnt_bodyid[joints[0]])
    terminal = int(model.jnt_bodyid[joints[-1]])
    path = set(_ancestors(model, terminal))
    candidates = [body for body in _descendants(model, root)
                  if _side(model.body(body).name or "") in (None, side)]
    keywords = {
        "shoulder": (("lower_shoulder", 50), ("shoulder", 40)),
        "elbow": (("lower_elbow", 50), ("elbow", 40)),
        "wrist": (("tool0", 70), ("tcp", 65), ("tool", 60), ("ee", 55),
                  ("hand", 50), ("wrist", 40), ("gripper", 30)),
    }[semantic]
    scored = []
    depths = _depths(model)
    for body in candidates:
        name = (model.body(body).name or "").lower()
        score = max((score for word, score in keywords if word in name), default=0)
        if score:
            # Shoulder/elbow must lie on the actuated path; wrist may be a fixed child.
            if semantic != "wrist" and body not in path:
                continue
            # Prefer the semantic body itself over deeper camera/sensor/gripper
            # descendants which merely contain the same word in their name.
            scored.append((score, -depths[body], body))
    if scored:
        _, _, selected = max(scored)
        return model.body(selected).name
    index = {"shoulder": min(1, len(joints)-1),
             "elbow": max(1, (len(joints)-1)//2),
             "wrist": len(joints)-1}[semantic]
    selected = int(model.jnt_bodyid[joints[index]])
    warnings.append(f"{side}_{semantic} inferred geometrically as {model.body(selected).name}; verify this Body")
    return model.body(selected).name


def _base_body(model, left_joints, right_joints, explicit):
    if explicit:
        try:
            return model.body(explicit).name
        except KeyError as error:
            raise ValueError(f"Unknown base body: {explicit}") from error
    left = _ancestors(model, int(model.jnt_bodyid[left_joints[0]]))
    right = set(_ancestors(model, int(model.jnt_bodyid[right_joints[0]])))
    common = [body for body in left if body in right and body != 0]
    if not common:
        raise ValueError("The two arms have no named common base Body; use --base-body")
    named = [(body, (model.body(body).name or "").lower()) for body in common]
    for exact in ("base", "base_link", "torso"):
        for body, name in reversed(named):
            if name == exact:
                return model.body(body).name
    return model.body(common[0]).name


def _collision_groups(model, warnings):
    groups = sorted({int(model.geom_group[geom]) for geom in range(model.ngeom)
                     if model.geom_contype[geom] or model.geom_conaffinity[geom]})
    if len(groups) == 1:
        return groups
    if groups:
        warnings.append(f"multiple collidable geom groups found {groups}; collision guard will inspect all geometry until reviewed")
    else:
        warnings.append("no contact-enabled geoms found; collision geometry must be reviewed")
    return None


def generate_profile(model_path, project_root, robot, *, left_joints=None,
                     right_joints=None, base_body=None, frames=None,
                     forward_axis="+x"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", robot):
        raise ValueError("robot must contain only letters, numbers, underscores or hyphens")
    if forward_axis not in ("+x", "-x", "+y", "-y"):
        raise ValueError("forward_axis must be +x, -x, +y or -y")
    model_path = Path(model_path).resolve()
    project_root = Path(project_root).resolve()
    model = load_robot_model(model_path)
    warnings = ["initial_qpos uses model qpos0; preview and save a suitable pose with edit_initial_qpos.py"]
    resolved = {
        "left": _resolve_joints(model, left_joints, "left"),
        "right": _resolve_joints(model, right_joints, "right"),
    }
    overlap = set(resolved["left"]) & set(resolved["right"])
    if overlap:
        raise ValueError("Left and right arm joint lists overlap")
    frame_overrides = frames or {}
    result_frames = {"base": _base_body(model, resolved["left"], resolved["right"], base_body)}
    for side in SIDES:
        for semantic in ("shoulder", "elbow", "wrist"):
            key = side+"_"+semantic
            result_frames[key] = _semantic_body(
                model, side, semantic, resolved[side], frame_overrides.get(key), warnings)
    try:
        relative = model_path.relative_to(project_root).as_posix()
    except ValueError:
        relative = str(model_path)
        warnings.append("model is outside the project; generated profile contains an absolute path")
    profile = {
        "schema_version": 1,
        "robot": robot,
        "model_xml": relative,
        "frames": result_frames,
        "arms": {side: {"joint_names": [_joint_name(model, joint) for joint in resolved[side]]}
                 for side in SIDES},
        "initial_qpos": [float(value) for value in model.qpos0],
        "editor": {"forward_axis": forward_axis},
        "runtime": {"redis_key": "action_"+robot, "command_output": "arm_joints"},
    }
    groups = _collision_groups(model, warnings)
    if groups is not None:
        profile["ik"] = {"collision": {"geom_groups": groups}}
    report = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "all_joint_names": [model.joint(joint).name for joint in range(model.njnt)],
        "all_body_names": [model.body(body).name for body in range(1, model.nbody)],
        "arm_dofs": {side: len(resolved[side]) for side in SIDES},
    }
    return GeneratedProfile(profile, warnings, report)
