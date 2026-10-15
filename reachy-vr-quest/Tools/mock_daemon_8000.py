#!/usr/bin/env python3
"""Minimal HTTP/WS stand-in for reachy-mini-daemon on port 8000 (pose only).

Use when Quest teleop is configured but the real daemon is not installed yet.
Logs FullBodyTarget JSON from Quest and exposes last target via /api/state/full_state.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Reachy Mini (mock)", version="0-mock")

_last_target: dict[str, Any] | None = None
_last_at: float = 0.0
_msg_count = 0


def _summarize(payload: dict[str, Any]) -> str:
    """One readable line of every field, with the 4x4 head pose decoded."""
    parts: list[str] = []
    pose = payload.get("target_head_pose") or {}
    matrix = pose.get("m") if isinstance(pose, dict) else None
    if isinstance(matrix, list) and len(matrix) == 16:
        x, y, z = matrix[3], matrix[7], matrix[11]
        r00, r01, r02 = matrix[0], matrix[1], matrix[2]
        r10, r11, r12 = matrix[4], matrix[5], matrix[6]
        r20, r21, r22 = matrix[8], matrix[9], matrix[10]
        sy = math.sqrt(r00 * r00 + r10 * r10)
        if sy > 1e-6:
            yaw = math.atan2(r10, r00)
            pitch = math.atan2(-r20, sy)
            roll = math.atan2(r21, r22)
        else:
            yaw = math.atan2(-r01, r11)
            pitch = math.atan2(-r20, sy)
            roll = 0.0
        parts.append(
            "head "
            f"pos=({x:.4f}, {y:.4f}, {z:.4f}) "
            f"rpy_deg=({math.degrees(roll):.2f}, {math.degrees(pitch):.2f}, {math.degrees(yaw):.2f})"
        )
    if "target_antennas" in payload:
        parts.append(f"antennas={payload['target_antennas']}")
    if "target_body_yaw" in payload:
        yaw = float(payload["target_body_yaw"])
        parts.append(f"body_yaw_rad={yaw:.4f} body_yaw_deg={math.degrees(yaw):.2f}")
    arms = payload.get("arms")
    if isinstance(arms, dict):
        for side in ("left", "right"):
            side_data = arms.get(side)
            if not isinstance(side_data, dict):
                continue
            joints = []
            for name in ("shoulder", "upper_arm", "lower_arm", "wrist"):
                joint = side_data.get(name)
                if not isinstance(joint, dict):
                    continue
                position = joint.get("position") or {}
                joints.append(
                    f"{name}=({float(position.get('x', 0)):.3f},"
                    f"{float(position.get('y', 0)):.3f},"
                    f"{float(position.get('z', 0)):.3f})"
                )
            if joints:
                parts.append(f"{side}[{' '.join(joints)}]")
    controllers = payload.get("controllers")
    if isinstance(controllers, dict):
        for side in ("left", "right"):
            hand = controllers.get(side)
            if not isinstance(hand, dict):
                continue
            stick = hand.get("thumbstick") or {}
            pressed = [
                name
                for name in ("primary", "secondary", "thumbstick_click", "menu")
                if hand.get(name)
            ]
            parts.append(
                f"{side}_ctrl "
                f"trigger={float(hand.get('trigger', 0)):.2f} "
                f"grip={float(hand.get('grip', 0)):.2f} "
                f"stick=({float(stick.get('x', 0)):.2f},{float(stick.get('y', 0)):.2f}) "
                f"buttons={pressed or '-'}"
            )
    known = {"target_head_pose", "target_antennas", "target_body_yaw", "arms", "controllers"}
    for key in payload:
        if key not in known:
            parts.append(f"{key}={payload[key]!r}")
    return " | ".join(parts) if parts else json.dumps(payload)


@app.get("/api/state/full_state")
def full_state(
    with_target_head_pose: bool = False,
    with_target_body_yaw: bool = False,
    with_target_antenna_positions: bool = False,
    use_pose_matrix: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {"mock": True, "message_count": _msg_count, "last_at": _last_at}
    if _last_target is None:
        out["status"] = "no_quest_data_yet"
        return out
    if with_target_head_pose and "target_head_pose" in _last_target:
        out["target_head_pose"] = _last_target["target_head_pose"]
    if with_target_body_yaw and "target_body_yaw" in _last_target:
        out["target_body_yaw"] = _last_target["target_body_yaw"]
    if with_target_antenna_positions and "target_antennas" in _last_target:
        out["target_antenna_positions"] = _last_target["target_antennas"]
    if "arms" in _last_target:
        out["arms"] = _last_target["arms"]
    if "controllers" in _last_target:
        out["controllers"] = _last_target["controllers"]
    return out


@app.websocket("/api/move/ws/set_target")
async def ws_set_target(websocket: WebSocket) -> None:
    global _last_target, _last_at, _msg_count
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                _last_target = payload
                _last_at = time.time()
                _msg_count += 1
                if _msg_count == 1:
                    print("[mock_daemon] keys:", sorted(payload.keys()))
                    print("[mock_daemon] full payload #1:")
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                if _msg_count <= 3 or _msg_count % 30 == 0:
                    print(f"[mock_daemon] #{_msg_count} {_summarize(payload)}")
                await websocket.send_text(json.dumps({"status": "ok"}))
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"status": "error", "detail": "invalid json"})
                )
    except WebSocketDisconnect:
        print("[mock_daemon] Quest disconnected")


if __name__ == "__main__":
    import uvicorn

    print("[mock_daemon] http://0.0.0.0:8000/docs  ws set_target on /api/move/ws/set_target")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
