# DualArmPoseTeleop

基于 MuJoCo + Mink 的通用双臂位姿遥操。目前只附带 **Baxter** 开源模型；控制算法不按机器人名称分支。项目自包含，不依赖旧 GMR 目录。

## 快速开始

Python 3.10+，MuJoCo 3.3.1+（外观薄片网格需要 mesh surface inertia）：

```bash
python -m pip install -e .
python run.py --print-config
python edit_initial_qpos.py
python check_collisions.py
python run.py --simulation
```

实时输入仍需自行安装 `xrobotoolkit_sdk` 并启动对应服务。编辑器、碰撞审计、配置打印和单元测试不连接该 SDK。

默认仿真模式不读取实机反馈，但仍可能发布 Redis。测试时隔离下游实机接收端。只有 `--replay` 会禁用全部 SDK 和 Redis 输出。

## 日常只改 8 项

`config/teleop.json` 保留机器人、仿真模式、映射模式、Redis 地址、人体大小臂长度、最大关节速度和碰撞检测开关。

- `config/devices.json`：三个 Tracker 序列号，已保留原配置。
- `robots/baxter.json`：机器人专属配置与 initial_qpos。
- `src/dual_arm_pose_teleop/defaults.py`：高级默认值，一般不用改。
- `python run.py --print-config`：打印实际生效配置及来源，不连接设备。

高级覆盖放在 teleop.json 的 `advanced` 中，例如 `"advanced": {"ik": {"iterations": 15}}`。优先级：默认值 → 设备/机器人 → 日常配置 → advanced → 命令行。旧完整 JSON 仍可通过 `--config` 加载；指定 --config 时不叠加日常配置且忽略 --robot。

## 从 URDF 一键生成 XML 和配置

对于已经展开的标准 URDF，使用统一导入命令：

```bash
python tools/import_robot.py /path/to/my_robot.urdf --robot my_robot
```

成功后会同时生成：

```text
assets/my_robot/my_robot.xml
assets/my_robot/meshes/
assets/my_robot/source/my_robot.urdf
assets/my_robot/import_report.json
robots/my_robot.json
```

导入器保留 URDF 关节轴、安装变换、限位和惯量；外观与碰撞分别放入 MuJoCo geom group 2 和 3，并将引用的 OBJ、二进制 STL 或 MSH 网格复制到项目内。`package://` 路径无法从 URDF 所在目录找到时，可重复提供 ROS 包根目录：

```bash
python tools/import_robot.py /path/to/my_robot.urdf --robot my_robot \
  --package-root /path/to/catkin_ws/src
```

工具会继续自动生成左右臂配置。关节命名不符合 `left/right` 或运动链存在歧义时，可使用下节相同的 `--left-joints`、`--right-joints` 和 Body 参数显式指定。

输入必须是已经展开的 `.urdf`，Xacro 先执行 `xacro robot.urdf.xacro -o robot.urdf`。OBJ、STL、MSH 可直接复制；常见 DAE/PLY/GLB 等格式安装可选依赖后会自动转为 OBJ：

```bash
python -m pip install -e ".[urdf-import]"
```

活动 Link 缺少惯量、mimic 关节和不受支持的关节类型会拒绝导入，而不会静默编造动力学或耦合关系。`--force` 会替换同名的已生成模型与配置，使用前应确认名称正确。

完成导入后只需检查和微调：

```bash
python edit_initial_qpos.py --robot my_robot
python check_collisions.py --robot my_robot
python run.py --robot my_robot
```

`initial_qpos`、肩肘腕语义点、末端 TCP、基座前向轴与碰撞几何仍必须目视确认。URDF 没有表达环境碰撞、控制器安装外参或真实夹具 TCP，这些内容无法仅从文件可靠推断。

## 适配已有 MuJoCo XML

可以先自动生成候选配置：

```bash
python tools/generate_robot_config.py assets/my_robot/my_robot.xml --robot my_robot
```

名称中能识别 `left/right`（或 `l_/r_`）并且每侧是一条串联运动链时，工具会自动提取左右关节、语义 Body、完整 `qpos0` 和碰撞几何组，写入 `robots/my_robot.json`。它不会覆盖已有文件，确实需要覆盖时显式加 `--force`；只想预览可加 `--stdout`。

名称不规范或分支存在歧义时，工具会拒绝猜测。此时使用 `--left-joints`、`--right-joints`、`--base-body` 和六个 `--left/right-shoulder/elbow/wrist-body` 参数明确指定，例如：

```bash
python tools/generate_robot_config.py assets/my_robot/my_robot.xml --robot my_robot \
  --left-joints left_j1,left_j2,left_j3,left_j4,left_j5,left_j6 \
  --right-joints right_j1,right_j2,right_j3,right_j4,right_j5,right_j6 \
  --base-body base_link \
  --left-shoulder-body left_link2 --left-elbow-body left_link4 --left-wrist-body left_tool0 \
  --right-shoulder-body right_link2 --right-elbow-body right_link4 --right-wrist-body right_tool0 \
  --forward-axis +x
```

生成器只创建候选 JSON，不修改 `config/teleop.json`，也不进行实机通信。`qpos0` 通常不是适合遥操的初始姿态，生成后仍要用编辑器确认关键点和初始姿态，再执行碰撞审计。

准备好可正常加载的 XML 后，在 `robots/<name>.json` 提供以下内容即可使用同一套控制流程：

1. `schema_version: 1`、robot 名称和 model_xml 相对项目路径。
2. frames：基座、左右肩部、肘部、末端的 **body 名称**，共七项。
3. arms：左右机械臂按控制顺序排列的关节名称。
4. initial_qpos：完整模型的初始关节角，由编辑器预览、调整并保存。
5. 可选 editor.forward_axis：基座坐标系前向轴；Baxter 为 +x。

```bash
python run.py --robot <name>
python edit_initial_qpos.py --robot <name>
python check_collisions.py --robot <name>
```

求解器自动读取关节限位、qpos 地址和非手臂自由度，并从 FK 取得肩肘腕几何；不需要逐 Link 填写旋转 offset。非手臂关节锁定在 initial_qpos。肩部不创建 IK 任务，肘部是位置次级目标，末端是位置和姿态目标。

模型必须保留语义 body 名称，两个手臂关节列表不能重叠。肩肘腕之间的两连杆计算是映射近似，不是任意机械结构的严格可达性证明；最终受完整模型 IK、限位和碰撞约束。增加机器人需要验证模型和关键点语义，但不应修改控制算法。

模型准备与遥操运行分开：已有 MJCF 时无需运行转换脚本。`tools/build_baxter.py` 仅用于复现本项目的官方 Baxter URDF 转换，不是任意 URDF 的通用导入器。

## Baxter 模型

来源：[RethinkRobotics/baxter_common](https://github.com/RethinkRobotics/baxter_common)，BSD-3-Clause，固定提交：
`6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e`。

```text
assets/baxter/baxter.xml                           运行模型
assets/baxter/meshes/                              转换后的 OBJ 外观/碰撞网格
assets/baxter/source/baxter_description/urdf/baxter.urdf  官方原始 URDF
assets/baxter/source/baxter_description/meshes/     原始 DAE 网格
assets/baxter/LICENSE                              上游许可证
assets/baxter/provenance.json                      固定版本与来源校验
```

原始 URDF 是官方保留的预展开兼容版本（上游注释标记 deprecated，建议 ROS 用户用 xacro）。运行不需要 ROS/xacro。关节轴、安装变换、限位和惯性由该 URDF 转换；薄片外观使用 shell inertia，body 惯性仍采用原始 URDF 明确给出的值。当前末端为 hand 法兰坐标系，没有额外安装实体夹爪，不等同于含夹爪 TCP 的模型。

坐标系：+X 前、+Y 左、+Z 上。模型共 15 个自由度：head_pan 与左右各 7 个手臂关节。头部锁定为 0。XML 的 qpos 顺序是头部、右臂、左臂；对外手臂输出始终是左臂、右臂，靠关节名称映射，不使用固定切片。

## 初始位姿与标定

`python edit_initial_qpos.py` 支持滑条、数值输入、左右镜像和生成 L 型。界面旋转角单位是度，JSON 内单位是弧度。Baxter 默认前向 +x；关节限位不能满足严格 L 型时允许配置限定的近似姿态，仍报告实际倾角和误差，不修改物理限位。预览后点击 Save to config，仅原子更新机器人配置的 initial_qpos。

按此前约定，编辑器生成、镜像和保存不检测碰撞；保存不等于安全。调整后运行 `python check_collisions.py`。运行时碰撞保护仍开启。

- A：遥操/暂停；每次进程启动首次进入前需要会话零点采样。
- 长按 Y：重新标定；暂停时标完仍保持暂停。
- X：退出。

腰部朝前，双臂保持与机器人初始姿态相对应的姿势，静止约 1 秒。标定期间不发布关节目标；静止性或新帧数量不满足会重新采样。人体外参、佩戴方向需要实际确认，机器人几何不需要人为猜测。

每次启动采集会话零点，避免重新佩戴/VR 重定位污染旧标定。Baxter 使用独立的 `calibration/baxter_default.json`，配置指纹变化会使旧标定失效，不复用之前机器人的标定。

## 映射与调试

默认 `cooperative_wrist`：双手使用共同的身体到机器人旋转和统一臂长比例，保持双手相对运动关系；肘姿态提供翻肘偏好。目标超出映射可达域时保持上一对目标，不单独缩小夹持距离。它不是刚性闭链控制，实际 IK 仍可能存在误差。

`segment_direction`：大小臂方向生成肘腕目标。运行期只使用头显位置、腰部姿态、肘部姿态及手柄位姿；腰部/肘部位置不持续参与目标计算。头显姿态不控制身体朝向。

MuJoCo 调试点：左目标青色、右目标橙色、实际点白色、误差线红色。日志位于 `logs/teleop_<robot>_<time>.jsonl`，包含 SDK、映射、IK、Redis 耗时及异常前序帧。

```bash
python run.py --simulation --record recordings/session_01.jsonl
python run.py --replay recordings/session_01.jsonl
python -m unittest discover -s tests -v
```

录制文件必须不存在；回放禁止 SDK/Redis，按记录时间播放，不覆盖标定文件。测试不等于实际设备延迟/安全验证。

## Redis 协议

保留事务 pipeline 批量发送及按键数据。Baxter 输出键改为 `action_baxter`，避免继续向旧机器人动作键写入不同语义的关节角：

```text
controller_data       左右手柄按键、扳机、摇杆和时间戳
action_baxter         14 维手臂角度，单位 rad
t_action              真正发布动作时更新的毫秒时间戳
teleop_command_valid  0/1，有效标志默认 200 ms TTL
```

14 维顺序：
`left_s0,s1,e0,e1,w0,w1,w2, right_s0,s1,e0,e1,w0,w1,w2`（各项均带 left_/right_ 前缀）。
读入 `real_robot_joint_state` 的顺序相同。真机模式必须提供新鲜反馈，推荐单键 JSON：
`{"positions": [...], "timestamp_ms": ...}`；也兼容数组与 `t_real_robot_joint_state` 配套时间戳。仿真完全忽略实测反馈。

接收端必须同步更新键名和关节映射，检查有效标志及时间戳；有效标志缺失或非 1 时保持当前位置。发布失败、数据停更、碰撞或明显 IK 误差会保持/暂停，不刷新动作时间戳。输出限速、加速度约束与暂停恢复渐入仍保留。该项目不是 Baxter 实机 SDK 驱动。

## 通用碰撞几何约定

Baxter XML 将外观放在 geom group 2、专用碰撞几何放在 group 3，机器人配置 `ik.collision.geom_groups: [3]` 显式选择后者。这与 contype/conaffinity 不同：contact mask 不会自动禁用主动距离检查。已有模型也可选择自己的分组；未指定分组时保留检查全部几何的行为。

运行时检查同臂非相邻部件、双臂之间、手臂与模型基座/环境之间的距离；IK 和限速后的关节路径都检查。安全距离默认 20 mm，约束激活距离 80 mm。路径超出采样预算会拒绝，不降低采样密度。

专用碰撞几何仍是模型近似，网格采用凸碰撞表示。不得用关闭碰撞检测或自动忽略初始姿态中的所有接触来掩盖错误。离散路径检测不是连续碰撞检测，模型中没有的桌面、人员和物体不可检测，不能替代实机安全系统。

Baxter 的 XML 内附原生 `contact/exclude` 装配规则，来自 Rethink Robotics 维护的 [MoveIt 语义配置](https://github.com/moveit/moveit_robots/blob/79c488b4afa548e19256f7adfee4fe1e7801f91f/baxter/baxter_moveit_config/config/baxter_base.srdf.xacro)（BSD；原文件与 package.xml 一并保留）。只继承同臂装配及肩部安装区规则，不继承跨臂、前臂/末端对基座的排除。程序统一读取 XML 的明确排除规则；配置中无需重复列出。`assembly_exclusions.json` 是来源审计清单，不是运行参数。

附带初始姿态：上臂向外倾约 30°，前臂朝前，手腕滚转为 0。它是对称近似 L 型，不是上臂严格竖直；受到 Baxter 原始关节限位约束。编辑器可以重新调整，调整后必须重新做碰撞审计。离线截图可用 `python tools/render_robot.py preview.png`（此可选工具另需 Pillow）。
