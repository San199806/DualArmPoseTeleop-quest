"""Offline pose mirroring for the initial-qpos editor; no SDK or Redis."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class SymmetryResult:
    qpos: np.ndarray
    success: bool
    position_errors_m: dict
    orientation_error_deg: float
    message: str


def mirror_position(position, center, normal):
    delta = np.asarray(position)-center
    return np.asarray(position)-2*np.dot(delta, normal)*normal


def generate_l_pose(model, qpos, config, forward_axis="+y", collision_check=None):
    """Generate from a deterministic profile seed, then mirror; no external IO.

    Directions are expressed in the configured base frame. Wrist twist is a
    soft preference from the profile, not a guessed universal tool frame.
    """
    original = np.asarray(qpos, dtype=float).copy()
    if original.shape != (model.nq,) or not np.all(np.isfinite(original)):
        raise ValueError("Invalid initial pose")
    axes = {"+x": [1,0,0], "-x": [-1,0,0], "+y": [0,1,0], "-y": [0,-1,0]}
    if forward_axis not in axes:
        raise ValueError("Choose a horizontal base-frame forward axis")
    editor = config.get('editor', {})
    max_tilt = float(editor.get('max_upper_tilt_deg', 20))
    max_bend = float(editor.get('max_bend_deviation_deg', 20))
    if not (0 < max_tilt <= 45 and 0 < max_bend <= 45):
        raise ValueError('L-pose approximation limits must be in (0,45] degrees')
    seed = np.asarray(config["initial_qpos"], dtype=float)
    if seed.shape != original.shape or not np.all(np.isfinite(seed)):
        raise ValueError("Invalid profile initial_qpos")
    candidate = original.copy()
    for side in ("left", "right"):
        for name in config["arms"][side]["joint_names"]:
            j = model.joint(name).id
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("L-pose generation currently requires hinge arm joints")
            candidate[model.jnt_qposadr[j]] = seed[model.jnt_qposadr[j]]
    data = mujoco.MjData(model)
    data.qpos[:] = candidate
    mujoco.mj_forward(model, data)
    base_rotation = data.xmat[model.body(config["frames"]["base"]).id].reshape(3,3).copy()
    down = -base_rotation[:,2]
    forward = base_rotation @ np.asarray(axes[forward_axis])
    anchors = [data.xanchor[model.joint(config["arms"][side]["joint_names"][0]).id].copy()
               for side in ("left", "right")]
    lateral = anchors[0]-anchors[1]
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Coincident arm mounting anchors")
    lateral /= np.linalg.norm(lateral)
    if abs(np.dot(lateral,forward)) > .1 or abs(np.dot(lateral,down)) > .1:
        return SymmetryResult(original,False,{},0,
                              "Selected forward axis is incompatible with the shoulder symmetry plane; choose another axis")
    bodies = [model.body(config["frames"]["left_"+key]).id for key in ("shoulder", "elbow", "wrist")]
    s,e,w = [data.xpos[b].copy() for b in bodies]
    upper, forearm = np.linalg.norm(e-s), np.linalg.norm(w-e)
    if min(upper, forearm) < 1e-5:
        raise ValueError("Invalid configured arm segment lengths")
    reference_rotation = data.xmat[bodies[2]].reshape(3,3).copy()
    joints = [model.joint(n).id for n in config["arms"]["left"]["joint_names"]]
    moving = [int(model.jnt_qposadr[j]) for j in joints]
    preference = candidate[moving].copy()
    bounds = [model.jnt_range[j] if model.jnt_limited[j] else
              (preference[i]-2*np.pi, preference[i]+2*np.pi) for i,j in enumerate(joints)]
    low,high = np.asarray(bounds).T
    # Keep generated poses away from hard stops, including the mirrored arm.
    for i,j in enumerate(joints):
        if model.jnt_limited[j]:
            margin = min(np.deg2rad(2), (high[i]-low[i])*.1)
            low[i] += margin
            high[i] -= margin
    if np.any(low >= high):
        raise ValueError("Invalid arm joint limits")

    upper_goal = down.copy()
    def evaluate(x):
        data.qpos[:] = candidate
        data.qpos[moving] = x
        mujoco.mj_forward(model,data)
        s,e,w = [data.xpos[b].copy() for b in bodies]
        return e-s-upper*upper_goal, w-e-forearm*forward

    def residual(x):
        a,b = evaluate(x)
        rot = Rotation.from_matrix(reference_rotation @ data.xmat[bodies[2]].reshape(3,3).T).as_rotvec()
        return np.r_[a/.002, b/.002, rot*.05, (x-preference)*.01]

    rng = np.random.default_rng(0)
    seeds = [preference, model.qpos0[moving]]
    seeds.extend(preference+rng.normal(0,1,len(moving)) for _ in range(6))
    best = None
    for start in seeds:
        result = least_squares(residual,np.clip(start,low+1e-10,high-1e-10),
                               bounds=(low,high),max_nfev=250)
        errors = [float(np.linalg.norm(v)) for v in evaluate(result.x)]
        score = max(errors)
        if best is None or score < best[0]:
            best = score,result.x.copy()
        if score < .001:
            break
    error,solution = best
    # An unreachable vertical upper arm is a shape preference failure, not
    # permission to accept inaccurate IK. Mirror the actual feasible pose.
    evaluate(solution)
    s,e,w = [data.xpos[b].copy() for b in bodies]
    def angle(a,b):
        return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/
                     (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
    tilt = angle(e-s,down)
    forward_error = angle(w-e,forward)
    bend = angle(e-s,w-e)
    approximate = error > .005
    if tilt > max_tilt or forward_error > 10 or abs(bend-90) > max_bend:
        return SymmetryResult(original,False,{"L_shape":error},0,
                              f"No suitable L pose: upper tilt={tilt:.2f} deg, "
                              f"forearm deviation={forward_error:.2f} deg, bend={bend:.2f} deg")
    candidate[moving] = solution
    mirrored = symmetrize_pose(model,candidate,config,"left",joint_margin_deg=2)
    collision_report = collision_check(mirrored.qpos) if mirrored.success and collision_check else None
    # Unequal mounting heights/angles can make the nearest LEFT pose
    # unreachable on the right. Search slightly more tilted feasible poses,
    # without relaxing the mirror task tolerances or either arm's bounds.
    if not mirrored.success or (collision_report is not None and not collision_report.safe):
        direction = (e-s)/np.linalg.norm(e-s)
        tangent = direction-np.dot(direction,down)*down
        if np.linalg.norm(tangent) < 1e-5:
            tangent = lateral-np.dot(lateral,down)*down
        tangent /= np.linalg.norm(tangent)
        for extra in (2,4,6,8,10,12):
            requested_tilt = tilt+extra
            if requested_tilt > max_tilt:
                break
            upper_goal = (np.cos(np.deg2rad(requested_tilt))*down +
                          np.sin(np.deg2rad(requested_tilt))*tangent)
            fit = least_squares(residual,np.clip(solution,low+1e-10,high-1e-10),
                                bounds=(low,high),max_nfev=250)
            evaluate(fit.x)
            ps,pe,pw = [data.xpos[b].copy() for b in bodies]
            trial_tilt,trial_forward,trial_bend = angle(pe-ps,down),angle(pw-pe,forward),angle(pe-ps,pw-pe)
            if trial_tilt > max_tilt or trial_forward > 10 or abs(trial_bend-90) > max_bend:
                continue
            candidate[moving] = fit.x
            trial = symmetrize_pose(model,candidate,config,"left",joint_margin_deg=2)
            if trial.success:
                report = collision_check(trial.qpos) if collision_check else None
                if report is not None and not report.safe:
                    continue
                mirrored = trial
                collision_report = report
                tilt,forward_error,bend = trial_tilt,trial_forward,trial_bend
                approximate = True
                break
    if collision_report is not None and not collision_report.safe:
        mirrored.success = False
        mirrored.message = 'Collision rejected: '+collision_report.message
    if not mirrored.success:
        mirrored.qpos = original
    kind = "Approximate L pose" if approximate else "Strict L pose"
    mirrored.message = (f"{kind} ({forward_axis} forward): upper tilt={tilt:.2f} deg, "
                        f"forearm deviation={forward_error:.2f} deg, bend={bend:.2f} deg; " + mirrored.message)
    return mirrored


def _mirror_joint_signs(model, data, neutral, indices, source_side, target_side,
                        center, normal, wrist_ids):
    """Sign per joint so the target arm tracks the reflected source arm.

    Copied angles and a single global negation are not enough when the two
    URDF mounts flip only some joint axes. Probe each joint from the neutral
    pose and keep the sign that lands the wrist on its reflected target.
    """
    signs = np.ones(len(indices[source_side]))
    epsilon = 0.2
    for index in range(len(signs)):
        scored = []
        for sign in (1.0, -1.0):
            data.qpos[:] = neutral
            data.qpos[indices[source_side][index]] = neutral[indices[source_side][index]]+epsilon
            data.qpos[indices[target_side][index]] = neutral[indices[target_side][index]]+sign*epsilon
            mujoco.mj_forward(model, data)
            reflected = mirror_position(data.xpos[wrist_ids[source_side]], center, normal)
            scored.append((float(np.linalg.norm(data.xpos[wrist_ids[target_side]]-reflected)), sign))
        signs[index] = min(scored)[1]
    return signs


def mirror_orientation(rotation, source_reference, target_reference, normal):
    # Reflection itself has determinant -1. Reflect the rotation increment
    # on BOTH sides to retain a proper rotation, then apply the opposite
    # Link's neutral mounting frame (the two URDF frames need not match).
    reflection = np.eye(3)-2*np.outer(normal, normal)
    return reflection @ (rotation @ source_reference.T) @ reflection @ target_reference


def symmetrize_pose(model, qpos, config, source_side, position_tolerance_m=0.005,
                    orientation_tolerance_deg=2.0, joint_margin_deg=0.0):
    if source_side not in ("left", "right"):
        raise ValueError("source_side must be left or right")
    target_side = "right" if source_side == "left" else "left"
    original = np.asarray(qpos, dtype=float).copy()
    if original.shape != (model.nq,) or not np.all(np.isfinite(original)):
        raise ValueError("Invalid initial pose")
    joints = {}
    indices = {}
    for side in ("left", "right"):
        joints[side] = [model.joint(name).id for name in config["arms"][side]["joint_names"]]
        if not joints[side] or any(int(model.jnt_type[j]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in joints[side]):
            raise ValueError("Symmetry requires configured scalar arm joints")
        indices[side] = [int(model.jnt_qposadr[j]) for j in joints[side]]
    if set(indices["left"]) & set(indices["right"]):
        raise ValueError("Arm joint lists overlap")
    data = mujoco.MjData(model)
    neutral = original.copy()
    for side in ("left", "right"):
        neutral[indices[side]] = model.qpos0[indices[side]]
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    anchors = {s: data.xanchor[joints[s][0]].copy() for s in ("left", "right")}
    center = (anchors["left"]+anchors["right"])/2
    normal = anchors["left"]-anchors["right"]
    if np.linalg.norm(normal) < 1e-6:
        raise ValueError("Coincident arm mounting anchors; cannot infer a symmetry plane")
    normal /= np.linalg.norm(normal)
    bodies = {s: {key: model.body(config["frames"][f"{s}_{key}"]).id
                  for key in ("shoulder", "elbow", "wrist")} for s in ("left", "right")}
    refs = {s: data.xmat[bodies[s]["wrist"]].reshape(3,3).copy() for s in ("left", "right")}
    data.qpos[:] = original
    mujoco.mj_forward(model, data)
    targets = {key: mirror_position(data.xpos[body].copy(), center, normal)
               for key, body in bodies[source_side].items()}
    target_rotation = mirror_orientation(
        data.xmat[bodies[source_side]["wrist"]].reshape(3,3).copy(),
        refs[source_side], refs[target_side], normal)
    moving = indices[target_side]
    low, high = [], []
    for j, index in zip(joints[target_side], moving):
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                margin = min(np.deg2rad(joint_margin_deg), (hi-lo)*.1)
                lo, hi = lo+margin, hi-margin
        else:
            width = 2*np.pi if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else 1.0
            lo, hi = original[index]-width, original[index]+width
        if lo >= hi:
            raise ValueError("Invalid destination joint limits")
        low.append(lo)
        high.append(hi)
    low, high = np.array(low), np.array(high)

    def evaluate(x):
        data.qpos[:] = original
        data.qpos[moving] = x
        mujoco.mj_forward(model, data)
        pos = {key: data.xpos[body]-targets[key] for key, body in bodies[target_side].items()}
        rot = Rotation.from_matrix(target_rotation @ data.xmat[bodies[target_side]["wrist"]].reshape(3,3).T).as_rotvec()
        return pos, rot

    def residual(x):
        pos, rot = evaluate(x)
        return np.r_[pos["shoulder"]/.005, pos["elbow"]/.005,
                     pos["wrist"]/.005, rot/np.deg2rad(2)]

    seeds = [original[moving], neutral[moving]]
    if len(indices[source_side]) == len(moving):
        source = original[indices[source_side]]
        signs = _mirror_joint_signs(
            model, data, neutral, indices, source_side, target_side, center, normal,
            {side: bodies[side]["wrist"] for side in ("left", "right")})
        signed = neutral[moving]+signs*(source-neutral[indices[source_side]])
        seeds.insert(0, signed)
        seeds.extend([source, -source])
    rng = np.random.default_rng(0)
    seeds.extend(np.clip(original[moving]+rng.normal(0,.5,len(moving)),low,high) for _ in range(2))
    best = None
    for seed in seeds:
        result = least_squares(residual, np.clip(seed,low+1e-10,high-1e-10),
                               bounds=(low,high), max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        pos, rot = evaluate(result.x)
        errors = {k: float(np.linalg.norm(v)) for k,v in pos.items()}
        angle = float(np.degrees(np.linalg.norm(rot)))
        score = max(max(errors.values())/position_tolerance_m, angle/orientation_tolerance_deg)
        if best is None or score < best[0]:
            best = (score,result.x.copy(),errors,angle)
        if score <= 1:
            break
    score, solution, errors, angle = best
    candidate = original.copy()
    candidate[moving] = solution
    success = score <= 1
    message = (f"{source_side} -> {target_side}: shoulder={errors['shoulder']*1000:.2f} mm, "
               f"elbow={errors['elbow']*1000:.2f} mm, wrist={errors['wrist']*1000:.2f} mm, "
               f"orientation={angle:.2f} deg")
    return SymmetryResult(candidate if success else original, success, errors, angle, message)
