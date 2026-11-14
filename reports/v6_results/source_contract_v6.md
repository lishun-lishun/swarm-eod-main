# V6 源码输入、输出与开关契约

| 开关 | 默认 | 唯一生效位置 | 安全边界 |
|---|---:|---|---|
| `V6_RAY_AUDIT` | 0 | `SictRescueDrone._map_update` 后 | 只用合法位姿、LiDAR、官方 35 角度；不修改 `scan_f` |
| `V6_VIEW_PLAN` | 0 | `FinalCompetitionDrone._explore_target/_set_path_to/_explore` | 空载、无认领、GPS 有效、普通兴趣低；已知自由连通与当前首段验证后才替换探索路径 |
| `V6_ROUTE_AUDIT` | 0 | `FinalCompetitionDrone._carry` 后 | 只诊断，不改动作 |
| `V6_CONNECTED_REPAIR` | 0 | `FinalCompetitionDrone._v6_observe_carry_route` | 中心代价非零且路径空、GPS 有效、120 帧限频；通过 A*、LiDAR 和载荷外廓验证才提交 |
| `V6_TASK_POLICY` | 0 | `SictRescueDrone._choose_bomb` | 事件触发，旧 V3 像素租约与新帧 ETA 分开 |
| `V6_BRAKE_SHADOW` | 0 | `FinalCompetitionDrone.control` 输出后 | 空载模型只读比较，原最终安全层仍负责实际推力 |

V6 没有使用固定炸弹表、地图源文件、真实速度或真实位置作为控制输入。`config/zones_location.txt` 与真值墙体仅在 `tools/benchmark_fixed_zones.py` 的赛后独立评估中读取。GPS 失效时，携弹新路线无法证明起点与载荷包络的全局连通性，因此拒绝替换；这是目前 Map09 seed862 6 号机问题仍未解决的明确限制。

`v6_visibility.py` 的证据格记录射线通过时间，不是炸弹不存在的逻辑命题。`v6_connected_route.py` 的 `MAP_CONNECTED` 只证明已知自由栅格上的连通，不保证真实世界一定通过；当前 LiDAR 首段、机身凸包与完整载荷扫掠检查是提交新路线的附加条件。`v6_route_monitor.py` 的 `commanded_speed` 输入是归一化推力范数，仅表明存在移动意图，不作为物理速度估计。`v6_braking_audit.py` 仅适用于空载单轴满反推力，不能用于 PivotJoint 载荷。

官方逐图 JSON、配对结果与保留/拒绝结论见 `analysis.md` 和 `ablation.json`。未通过的主动候选保留源码与独立开关供复现，比赛默认仍为 V5 稳定动作链。
