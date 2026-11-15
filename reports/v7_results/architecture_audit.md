# V7 控制调用链与合法性审计

## 基点与入口

- 基点：`v6/evidence-based` 的 `498b56448a8a72b68092a5fd9250f5c65b1c3737`；开发分支：`v7/gateway-egotrail`。
- 比赛入口：`src/swarm_rescue/solutions/my_drone_eval.py` 创建 `FinalCompetitionDrone`，继承 `FusionRescueDrone` 和 `SictRescueDrone`。
- `SictRescueDrone._setup()` 在首个控制帧创建 `GridMap`、任务知识和状态。因此 V7 的 `GatewayPlanner` 在 `FinalCompetitionDrone._setup()` 调用父类之后初始化；构造函数不读取尚未存在的 `grid`。
- 每帧顺序为 `_update_pose` → `_perceive` → `_map_update` / 融合 → 原任务状态机 → `_explore` 或 `_carry` → `FinalCompetitionDrone._control_with_safety()`。V7 网关动作只在探索分支，V7 退路只在载弹分支。原抓取、运输、中心最后推送与最终安全层保留。

## 数据与开关

| 开关 | 默认 | 接入位置 | 当前结论 |
|---|---:|---|---|
| `V7_RAY_AUDIT` | 0 | `_map_update` | 只读记录真实 35 方向语义射线；网关影子/主动开关也自动启用 |
| `V7_GATEWAY_SHADOW` | 0 | `_explore` | 只规划、记录；不改变动作 |
| `V7_GATEWAY` | 0 | `_explore` | 单侦察机已知自由区路径；Map08 回归，拒绝默认启用 |
| `V7_PROBE` | 0 | 已到达网关后 | 3 帧 LiDAR 确认，单步最多 8px；Map08 仍掉弹，拒绝默认启用 |
| `V7_SWEEP` | 0 | 已到达网关后 | 35 射线三相位，官方主动测试尚未执行 |
| `V7_EGO_SHADOW` | 0 | `_carry` | 本机里程计轨迹与局部退路只读评估 |
| `V7_EGO_RETREAT` | 0 | `_carry` | Map09 seed862 单机行动无收益，拒绝默认启用 |

`V7_SCOUT_ID` 默认 9 是侦察机资源分配，不是地图或炸弹坐标。`V7_GATEWAY_START` 默认 900，可用于消融。`V7_EGO_TARGET_ID` 默认 -1；专项测试中设 6，只限制参与测试的无人机。V4/V5/V6 未验证的主动开关仍保持关闭。

## 坐标和安全不变量

- `GridMap` 原点为 `(-width/2, -height/2)`，栅格边长 `CELL=10`。Gateway 的目标及所有 A* 航点都位于 `occ<-0.6` 且净空至少 16px 的已知自由区，禁止对角穿角。起始接线同时检查当前 3 条临近 LiDAR 射线。导航中重新验证当前段与已知自由栅格，抵达须距网关小于 16px 且航点走完。
- Probe 只允许空载探索；连续 3 帧本机 LiDAR 清晰后单步不超过 8px。候选落在已知障碍或地图边界时拒绝，最终动作仍进入原安全链。Map06/08 的 Pymunk 实验已分别出现 8/3 次短探查实际抵达，但没有修复漏弹，Map08 官方处置从 8 降到 7，默认关闭。
- 语义射线继续使用现有 V6 几何记录：35 条实际方向、端点方向重复，射线间隔为 `2π/34`。LiDAR 可通行空间不会被当作语义检索完成，也不会把赛后真值送入控制器。
- `LocalOdomTrail` 复用 `_update_pose` 同一次 `odometer_values()` 读数；轨迹只在独立本机坐标系。不得用于 `GridMap`、无线共享、全局中心路由。GPS 无效、载弹、无实际路径进展、中心语义不近、没有动态无人机/旧脱困动作时才评估。当前局部验证使用完整抓取半径的保守圆包络；窄道通过率未标定，故主动功能默认关闭。
- 原载弹控制和最终安全层不使用空载线性动力学。稳定比赛 ZIP `artifacts/latest_upload/team006_final.zip` 的 SHA256 为 `02193D3B368AE2471FD163BDB090428FAEB02664DAFB3AD729F4141BD21FD6E3`，未替换。

## 独立评估器

`tools/benchmark_fixed_zones.py` 用 `config/zones_location.txt` 的固定炸弹坐标配置官方 Launcher，只在赛后审计中使用真值，并用 `DisposalCenter.activate` 的官方事件确定真实处置。`tools/run_fixed_suite.py` 每张地图运行十架无人机、十枚固定炸弹、无 GPS 与无通信区域。`tools/summarize_v7_results.py` 输出配对消融、逐弹离线生命周期和失败事件。规划网关数、实际到达数、抓取数与官方处置数分开统计。
