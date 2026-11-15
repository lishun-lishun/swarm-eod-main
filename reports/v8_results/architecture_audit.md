# V8 实际调用链与合法性审计

基点：`v7/gateway-egotrail` 提交 `3c15137b54a5d07b37294b3d6da20a90a458c80d`；分支 `v8/macro-route`。官方入口仍为 `src/swarm_rescue/solutions/my_drone_eval.py`，创建 `FinalCompetitionDrone`。`SictRescueDrone._setup()` 在第一帧创建 `GridMap`；V8 的 MacroPlanner 不持有网格引用，运行时仅读取当前 `occ/clearance/ev`。

## 每帧路径

`_update_pose()` 只读取一次合法 `odometer_values()`，同一读数交给 V8 本机路径续航；随后 `_perceive()`、LiDAR 融合与任务状态转换照旧。只有旧状态机进入空载 `_explore()`，且没有已有走廊任务、V6/V7 主动任务时，V8 Macro 才可在普通 `_explore()` 之前提出候选。`_fetch()`、`_carry()` 立即释放 V8 私有目标，不清除旧抓取或返航路径。载弹 `_carry()` 仍先运行原控制器，V8 Ego 只在明确开关与局部条件下可能给出名义动作；最终控制仍进入 `FinalCompetitionDrone._control_with_safety()`。

## 开关与数据边界

| 开关 | 默认 | 作用 |
|---|---:|---|
| `V8_MACRO_SHADOW` | 0 | 每至少 90 帧规划候选，仅记录目标、护航门槛、展开数与耗时 |
| `V8_MACRO_ACTIVE` | 0 | 单侦察机按已知自由区四邻接路线执行；Map08 真实掉弹，拒绝默认启用 |
| `V8_EGO_SHADOW` | 0 | 合法 GPS 锚定现有中心路径；无 GPS 时评估正向短段，不改动作 |
| `V8_EGO_ACTIVE` | 0 | 需有可信旧路径和实时载荷/LiDAR验证；Map09 6 号机不存在锚点，未启用 |

V8 的 `V8_SCOUT_ID` 默认 9 只是单机资源限制，不是地图或炸弹硬编码。`MissionPreservationGuard` 使用本机状态和本机收到的通信消息，字段 `step` 是通信接收帧；无通信时不假设队友消失。评估时从公开配置注入正确 `V5_TIME_HORIZON`，比赛控制器自身不读取隐藏地图/比赛真值。V8 所有模块源码来自用户给定 ZIP，未修改官方引擎、地图、传感器、炸弹布局、计分或原飞控。

## 可复现命令

```powershell
$env:PYTHONPATH='src'
& 'E:\envs\swarm_rescue\python.exe' -m pytest -q tests/test_v8.py
& 'E:\envs\swarm_rescue\python.exe' scripts/check_submission.py
$env:V8_MACRO_SHADOW='1'
& 'E:\envs\swarm_rescue\python.exe' tools/run_fixed_suite.py --label v8_macro_shadow_e861 --engine-seed 861 --maps Map06 Map08
$env:V8_MACRO_SHADOW='0'; $env:V8_MACRO_ACTIVE='1'; $env:V5_TIME_HORIZON='5400'
& 'E:\envs\swarm_rescue\python.exe' tools/run_fixed_suite.py --label v8_macro_active_e861 --engine-seed 861 --maps Map06 Map08
```

这些工具把固定坐标仅交给官方 Launcher。赛后真值审计在 `tools/audit_v8_macro_stances.py` 独立执行，结果绝不传回控制器。
