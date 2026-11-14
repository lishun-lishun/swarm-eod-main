# V6 控制链与失效审计

基点是 `v5/deadline-progress` 的 `896510d98a73a316da1100104b580f09dba6ede4`，远端 `origin/v5/deadline-progress` 已核对为同一提交。开发分支为 `v6/evidence-based`。官方入口 `my_drone_eval.py` 实例化 `FinalCompetitionDrone`，继承链为 `FinalCompetitionDrone → FusionRescueDrone → SictRescueDrone`。`FinalCompetitionDrone.control()` 先调用原主干与最终 LiDAR 安全层；V6 制动只读审计位于最终输出之后，绝不替换动作。

旧 `GridMap.integrate()` 的 `scan_f` 仍使用默认 181 方向代理。V6 在独立 `SemanticRayEvidence` 中按官方 `compute_ray_angles(2π,35)` 记录真实 35 条射线中心线；两个角度数组已逐元素核对一致。此记录只是合法射线经过证据，不作为无弹证明，也不改变旧占据图、广播或 `scan_f`。无 GPS 时只在短暂里程计可信窗口内累计全局证据；更长失效只保留原局部感知而不宣称全局覆盖。

Map06 seed861 的历史漏弹索引 2、3 在真值地图分别有 20、29 个可见站位；Map08 索引 0、3 分别有 78、93 个。赛后审计显示四枚对应的 V6 射线经过数均为 0，且每枚在十机最终已知自由地图上的可达真值站位最大数为 0。真值只在 Launcher 结束后由独立评估器读取。此结果表明可见站位存在，但当前地图和连通性没有把无人机带到那里；只降低语义覆盖代理的射线数不足以解决漏探。

携弹旧路径存在 `center_cum>0` 而 `path=[]` 的故障。V6 `RouteIntegrityMonitor` 区分无路线、航点停滞和地图中心假到达；它本身不触发逃逸动作。可选修复仅在已知自由 A* 连通、当前 LiDAR 首段及完整载荷摆动包络验证后事务式提交；无法验证时保留旧控制链。Map09 seed862 第 6 号机关键空路径发生在第 5010、5360 步，当时历史合法轨迹显示 GPS 无效；保守候选因此不能提交。Map09 seed862 和 Map07t4 seed861 均无验证成功的新路线，官方处置数保持基线。

V6 任务候选仅用合法已发现炸弹、当前已知路线和本机实际收到的租约。旧 V3 报价仍为像素，V6 ETA 为帧；前者只作为新鲜认领锁，不与后者比较。V6 事件策略只在任务集合、失败、处置、认领、名额或租约冲突变化时重评。Map04 seed861 虽保持 10/10，但官方分从 97.7 降至 94.0；Map09 seed862 仍为 9/10，故保持关闭。

七个包内 V6 模块先逐字节复制到正式策略目录；随后只对 `v6_route_monitor.py` 修复“航点切换未重置停滞窗口”的集成缺陷。原 `oriented_footprint.py` 未覆盖。随包 22 项独立测试先在原包运行，之后在正式导入路径运行并通过。所有主动开关默认 0；稳定比赛 ZIP 未改动。
