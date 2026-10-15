# DualArmPoseTeleop

MuJoCo + Mink 上的 Lycus 双臂遥操作。Quest 上半身骨架和手柄位姿从 `ws://0.0.0.0:8000/api/move/ws/set_target` 进入，映射后做双臂逆解。

## 启动

```bash
python -m pip install -e .
python run.py
```

`config/teleop.json` 的机器人是 `lycus`。仿真模式下忽略实机关节反馈。Quest 右柄 A 开始或暂停，按住左柄 Y 重新标定，左柄 X 退出。

标定要求腰朝前、双臂保持与机器人初始姿态对应的姿势，静止约 1 秒。每次进程启动都会重新采集这一姿态。

```bash
python run.py --print-config
python edit_initial_qpos.py
python check_collisions.py
python run.py --record recordings/session_01.jsonl
python run.py --replay recordings/session_01.jsonl
```

`--replay` 不连接 Quest，也不发布 Redis。其它运行仍可能向 `action_lycus` 发布 14 维手臂关节角。

## 配置

日常项在 `config/teleop.json`：仿真开关、映射模式、Redis 地址、人体臂长、最大关节速度、碰撞开关。机器人模型、关节和 `initial_qpos` 在 `robots/lycus.json`。其余默认值在 `src/dual_arm_pose_teleop/defaults.py`。

`cooperative_wrist` 用肩到手柄的笛卡尔方向，双手共用一个臂长比例。`segment_direction` 分别用上臂方向和前臂方向，再放到机器人的连杆长度上。

画面里左目标是青色，右目标是橙色，实际点是白色，误差是红线。

## 从 URDF 生成模型

```bash
python tools/import_robot.py /path/to/robot.urdf --robot my_robot
```

已有 MJCF 时用 `python tools/generate_robot_config.py assets/my_robot/my_robot.xml --robot my_robot`。生成后仍要用 `edit_initial_qpos.py` 确认初始姿态。

## Quest 工程

`reachy-vr-quest/` 是头显端 Unity 工程，向上面的 WebSocket 发送上半身骨架和手柄位姿。构建好的 APK 在 GitHub Release 里，不放进 git。
