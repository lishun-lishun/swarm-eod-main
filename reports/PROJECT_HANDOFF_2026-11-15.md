# Swarm-Rescue 本地项目交接文档（2026-11-15）

> 本文以当前磁盘、Git 分支、官方源码与已经保存的官方仿真 JSON 为准。固定炸弹坐标只供测试夹具与赛后独立评估使用，**不得写入比赛控制器**。

## 1. 一眼看懂当前状态

| 项目 | 当前事实 |
|---|---|
| 工程根目录 | `E:\code\final_fusai\swarm-eod-main` |
| Python / Conda 环境 | 环境名 `swarm_rescue`，解释器 `E:\envs\swarm_rescue\python.exe`；当前可导入 Pymunk 6.4.0 |
| 当前检出分支 | `main`，提交 `977afba038e327ee861e0bbef6311abfc690c084` |
| 当前工作树原有改动 | `reports/fixed_zones_work/plan_Map06.yml` 已修改；切换分支、清理或提交时不要误覆盖 |
| 已验证的稳定比赛包 | `artifacts/latest_upload/team006_final.zip`；SHA256 `02193d3b368ae2471fd163bdb090428faeb02664dafb3ad729f4141bd21fd6e3` |
| 稳定包中的策略 | ZIP 根目录 `my_drone_eval.py` → `blue_pid_explore/final_competition_drone.py` 的 `FinalCompetitionDrone` |
| 当前 `main` 工作树策略入口 | `src/swarm_rescue/solutions/my_drone_eval.py`，默认同样选择 `FinalCompetitionDrone` |
| 最新实验分支 | `v8/macro-route`（`26c8920`）；`v9/independent-controller`（`4dac144`，V9 默认关闭） |
| 已验证最佳固定布局成绩 | 稳定 ZIP、官方 Launcher/Pymunk、seed861：**93/100 枚、7/10 张满拆、平均官方分 83.78** |

**版本区别非常重要：**当前 `main` 工作树与稳定 ZIP 内有 4 个 Python 文件字节不同：`my_drone_eval.py`、`blue_pid_explore/final_competition_drone.py`、`legacy_sict_knowledge.py`、`pose_sict_drone.py`。所以“运行当前工作树”不等于“重跑已验证 ZIP”。稳定包通过了 `scripts/check_submission.py` 的 ZIP 检查；它的 93/100 是隔离安装该 ZIP 源码后的真实官方仿真结果，记录在 `v7/gateway-egotrail` 分支的 `reports/v7_results/submission_report.md` 和 `reports/fixed_suites/v7_packaged_all_e861/`。`main` 中另有 `artifacts/v7_delivery/team861_final.zip`，哈希与稳定包相同；不要把名称中的 `861` 误解为不同控制器。

## 2. 用户的最终要求与约束

1. **优先目标：**十张固定布局（Map01–Map09 加 Map07t4）都由 10 架无人机完成 **10/10 枚官方处置**。随后提高官方分。Map01 必须满拆，但不要求达到 92 分；其余每张地图目标是 **至少 92 分**，不能只靠平均分达标。
2. 策略必须有**跨地图、跨种子的泛化性**。不能把下表中的炸弹、墙、路口坐标写进控制器，也不能只针对某一张地图调整路线或盲搜参数。
3. 同时处理 **无 GPS/无 Compass 区域、无通信区域**，保留合法本机定位与失联任务恢复。无通信不能假定队友状态实时一致；无 GPS 时不能凭空制造可信世界坐标锚。
4. 以官方 Launcher、Pymunk、真实机身与 PivotJoint 载弹物理、相同固定布局和引擎种子做 A/B。单元测试通过、规划出路径、发现或抓取炸弹，都不能代替官方处置事件。
5. 保留旧稳定 ZIP、历史成绩和失败轨迹。任何原满拆地图掉弹、健康严重下降或超时，都应回滚或继续通用性修复；新算法未通过跨图验证不得替换提交包。
6. 允许修改的正式策略范围是 `src/swarm_rescue/solutions/`；独立测试、报告、工具可放在 `tests/`、`reports/`、`tools/`。不得改官方地图、炸弹配置、物理、传感器、计分或比赛模式；不得在 `control()` 读真值、固定布局或其他无人机的 Python 对象。赛后独立评估器可读真值做归因，但不能向控制器回传。
7. 需要交付可上传的完整源码与 ZIP、官方格式检查、逐图 JSON、健康与步骤、逐弹生命周期、失败原因、同种子消融，以及清楚的接收/拒绝结论。用户特别要求 `reports/` 随代码提交。

当前的开发主线是：先修复 Map06/Map08 的合法语义漏探与走廊/分支实际到达，再修复 Map07t4 的中心假到达及 Map08/Map09 的载弹返航和通道拥堵，最后优化 Map07/Map09 完成时刻；必须保护 Map01–05 和 Map07 等原已满拆地图。

## 3. 官方规则与环境事实

来源：`src/swarm_rescue/simulation/reporting/score_manager.py`、`time_score.py`、`simulation/gui_map/map_abstract.py`、`simulation/utils/constants.py`、`simulation/elements/sensor_disablers.py` 和各 `maps/map_*.py`。

### 救援评分

设 `N` 为官方实际处置数量（固定测试共 10 枚），`H` 为官方“返航健康分”百分比，`t` 为完成步数，`T` 为该图步数上限：

```text
官方分 = 0.6 × (100 × N/10) + 0.2 × H + 0.2 × 时间分
若 N < 10：时间分 = 0
若 N = 10：时间分 = 100 × clip((T - t)/(T - 0.3T), 0, 1)
```

探索率在蓝方救援总分的权重为 **0**；测试计划中的 `exploration_threshold_for_time_bonus: 90` 不应被理解为“探索到 90% 就有时间分”。**`mean_health_percent` 是诊断指标，不一定等于 `H`：有返航区时，只有返航区内无人机的健康计入官方健康分。** 只在炸弹与匹配的处置中心发生官方物理接触并触发 `DisposalCenter.activate` 时算一枚处置。

| 地图 | 控制帧上限 | 墙钟上限 |
|---|---:|---:|
| Map01 | 2000 | 90 秒 |
| Map02、Map03、Map04 | 5000 | 180 秒 |
| Map05 | 7200 | 240 秒 |
| Map06、Map07、Map07t4、Map08、Map09 | 5400 | 180 秒 |

每控制帧内 Pymunk 还有多个微步；策略不能把空载离散动力学套用到铰接载弹。官方 LiDAR 为 **181 方向、230px、360°**；语义传感器为 **35 方向、130px、360°**，返回的是命中实体的**变长列表**；通信半径 **180px**。`NO_GPS_ZONE` 同时禁用 GPS 与 Compass；`NO_COM_ZONE` 禁用 Communicator。固定布局测试工具对每图都启用这两类区域。正式策略只能使用官方合法传感器与实际收到的通信消息。

## 4. 十张固定地图的 100 枚炸弹坐标

唯一输入源：`config/zones_location.txt`。下列索引 `0…9` 按该文件每图的“Bomb added”顺序；单位为官方世界坐标 px，格式 `(x, y)`。`tools/benchmark_fixed_zones.py` 仅作为**测试夹具**读取这些点并传给 Launcher，正式无人机不得读取该文件。

| 地图 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Map01 | (-131,382) | (-105,385) | (-610,-382) | (270,-380) | (-97,-67) |
| Map02 | (135,90) | (-134,87) | (-140,-78) | (136,-78) | (579,430) |
| Map03 | (16.5,-125) | (-112.5,-124) | (317.5,357) | (-154.5,359) | (-416.5,90) |
| Map04 | (-644,-275) | (-816,-280) | (-407,97) | (-514,97) | (-640,95) |
| Map05 | (164,253) | (-172,-252) | (391,-500) | (499,-382) | (-22,358) |
| Map06 | (-543,-63) | (-519,-519) | (-504,-337) | (-449,-339) | (-694,109) |
| Map07 | (-664,-180) | (-423,-168) | (-217,-282) | (-650,92) | (-410,104) |
| Map07t4 | (662,180) | (640,-94) | (805,-313) | (810,-418) | (550,-540) |
| Map08 | (566,-535) | (-128,-428) | (-239,-533) | (-7,-533) | (509,-96) |
| Map09 | (-344,342) | (-656,341) | (608,530) | (409,532) | (49,472) |

| 地图 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| Map01 | (305,131) | (601,-383) | (604,381) | (270,381) | (451,-380) |
| Map02 | (-585,434) | (-491,337) | (485,335) | (506,-189) | (-505,-185) |
| Map03 | (-414.5,140) | (-143.5,119) | (-538.5,-161) | (522.5,-347) | (529.5,353) |
| Map04 | (-812,268) | (-818,462) | (809,-536) | (586,96) | (806,332) |
| Map05 | (23,-354) | (-506,397) | (-398,505) | (577,580) | (-583,585) |
| Map06 | (654,356) | (-297,529) | (-613,528) | (-822,-528) | (-831,-295) |
| Map07 | (-805,275) | (-814,360) | (-810,460) | (131,349) | (225,-272) |
| Map07t4 | (152,-538) | (502,-95) | (511,7) | (432,172) | (-125,-399) |
| Map08 | (649,-90) | (804,-49) | (672,281) | (207,288) | (427,173) |
| Map09 | (-230,472) | (-551,112) | (-687,-188) | (-707,530) | (823,528) |

## 5. 已做成什么样：稳定基线与 V8/V9 实验

### 已验证稳定 ZIP，固定布局 seed861

下表是**按 ZIP 内源码隔离安装后**，用官方 Launcher/Pymunk、10 机、固定十弹、无 GPS/无通信区实际得到的分数；`main` 工作树当前字节不同，不能无条件继承这些分数。

| 地图 | 实际处置 | 官方分 | 平均健康 | 完成/终止帧 |
|---|---:|---:|---:|---:|
| Map01 | 10/10 | 83.8 | 93.2% | 1639 |
| Map02 | 10/10 | 96.7 | 91.6% | 1781 |
| Map03 | 10/10 | 99.1 | 95.6% | 984 |
| Map04 | 10/10 | 97.7 | 98.4% | 1855 |
| Map05 | 10/10 | 98.5 | 92.4% | 1949 |
| Map06 | 7/10 | 58.1 | 80.4% | 5400 |
| Map07 | 10/10 | 89.0 | 91.2% | 3367 |
| Map07t4 | 8/10 | 64.1 | 80.4% | 5400 |
| Map08 | 8/10 | 65.0 | 84.8% | 5400 |
| Map09 | 10/10 | 85.8 | 86.0% | 3770 |
| **合计/均分** | **93/100；7 张满拆** | **83.78** | — | — |

V8 的四项主动/影子开关全关时，seed861/862/863 十图均为 **93/100**，平均官方分分别 **83.78 / 81.63 / 81.95**。seed862/863 均只有 6 张满拆，Map09 各为 9/10。V8 主动 Macro 在 Map08 seed861 使 8/10 降至 6/10；因此未进入稳定 ZIP。详情在 `v8/macro-route` 分支的 `reports/v8_results/`。

V9 是 `v9/independent-controller` 分支的独立 `V9RescueDrone(DroneAbstract)` 实验。入口用 `V9_CONTROLLER=1` 才选择 V9，默认仍为旧控制器。46 项包内测试、集成后 48 项测试和提交检查通过；官方单机单弹完成 1/1（第 155 步）。但固定十弹 seed861 的 Map01/06/07t4/08/09 分别为 **6/10、6/10、0/10、0/10、2/10**，远低于上表。V9 **已被拒绝作为比赛默认策略**；seed862/863 的 V9 十图完整测试尚未运行，不能编造。详细报告和失败 JSON 在该分支的 `reports/v9_results/`。

### 尚未解决的主要失败

- **Map06：**稳定基线有 2 枚未被合法语义发现，另有已抓未处置；走廊与观察位的实际到达不足。V9 的健康大幅下降，不能作为修复。
- **Map08：**稳定基线有 2 枚未发现；分支虽被检测，进入/通过未完成。增加主动探索曾打乱抓取与载弹交付，造成 8→6 或 8→7，说明探索与运输需要联动验证。
- **Map07t4：**稳定基线 8/10；携弹机出现地图路径代价为零但真实中心语义仍远、路径退化为单点的“假到达”。近场接触/局部绕行未被证实可解决。
- **Map09：**seed861 可 10/10 但分数 85.8、完成偏晚；seed862/863 为 9/10。无 GPS 时，V8 的局部退路无法获得可靠锚点；需要合法定位与载弹连通路径，而非强推或盲目脱困。
- **Map07：**seed861 满拆但仅 89.0 分、3367 帧；在健康不变的计算下需约 2800 帧内完成才有机会达到 92。Map09 类似目标约 2603 帧。以上是计分推导的时刻目标，不是现有算法成绩。

## 6. 源码、报告与比赛调用链

```text
官方 Launcher / rescue 模式
  → src/swarm_rescue/solutions/my_drone_eval.py
  → FinalCompetitionDrone
  → src/swarm_rescue/solutions/blue_pid_explore/
  → 官方 DroneAbstract 允许的传感器、通信、动作
  → 官方 Pymunk 物理 / DisposalCenter.activate / ScoreManager
```

当前 `main` 的 `src/swarm_rescue/solutions/my_drone_eval.py` 总是选择 `FinalCompetitionDrone`。V9 分支的同名入口才有 `V9_CONTROLLER` 环境变量选择；不要在 `main` 上设置该变量后误以为已运行 V9。最可靠的“当前最优比赛代码”是稳定 ZIP 内的 `my_drone_eval.py` 加 `blue_pid_explore/` 全部文件；单独复制工作树中的一个 `final_competition_drone.py` 不能重现已验证版本。

重要路径：

- 固定炸弹布局：`config/zones_location.txt`。
- 固定布局单图工具：`tools/benchmark_fixed_zones.py`；完整十图工具：`tools/run_fixed_suite.py`。
- ZIP 格式检查：`scripts/check_submission.py`。
- 主工作树历史基线：`reports/fixed_suites/baseline_20261114_e861/summary.json` 与同目录 `analysis_progress.md`。
- **真正按稳定 ZIP 源码隔离运行的证据与工具：**`v7/gateway-egotrail` 分支的 `reports/v7_results/submission_report.md`、`reports/fixed_suites/v7_packaged_all_e861/`、`tools/test_packaged_fixed_suite.py`。这些文件不在当前 `main` 工作树中。
- V8 三种子结果与消融：`v8/macro-route` 分支的 `reports/v8_results/`、`reports/fixed_suites/v8_default_all_e861/862/863/`。
- V9 独立源码与拒绝报告：`v9/independent-controller` 分支的 `src/swarm_rescue/solutions/v9_rescue/`、`reports/v9_results/V9_experiment_report.md`。

## 7. 如何启动、检查与复现

以下命令在 **PowerShell** 中运行。切换工作树会影响当前未提交的 `plan_Map06.yml`，建议用独立 Git worktree 或先安全保存个人改动；不要执行会覆盖本地成果的 `git reset --hard`。

### 环境与稳定 ZIP 检查

```powershell
Set-Location 'E:\code\final_fusai\swarm-eod-main'
$PY = 'E:\envs\swarm_rescue\python.exe'
$env:PYTHONPATH = 'src'
& $PY -c 'import sys,pymunk; print(sys.executable, pymunk.version)'
& $PY scripts/check_submission.py artifacts/latest_upload/team006_final.zip
Get-FileHash artifacts/latest_upload/team006_final.zip -Algorithm SHA256
```

也可用 `conda activate swarm_rescue`，但绝对解释器路径最不容易误用其他环境。`Submission check passed` 只证明 ZIP 格式/依赖要求，不证明比赛得分。

### 跑当前工作树的固定布局

```powershell
$env:PYTHONPATH = 'src'
$env:V6_EVENT_AUDIT = '1'   # 只在独立评估器记录真实处置事件
& $PY tools/run_fixed_suite.py --label handoff_main_e861 --engine-seed 861
```

默认测试十张图；结果在 `reports/fixed_suites/handoff_main_e861/summary.json` 与逐图 `MapXX_e861.json`。只跑重点图的例子：

```powershell
& $PY tools/run_fixed_suite.py --label handoff_key_e861 --engine-seed 861 --maps Map06 Map07t4 Map08 Map09
& $PY tools/benchmark_fixed_zones.py --map Map08 --engine-seed 862 --out reports/fixed_suites/handoff_map08_e862.json
```

如需打开可视窗口观察官方比赛，可在测试夹具已生成某图的 `plan_*.yml` 与 `bombs_*.json` 后启动 Launcher；这是直接运行**当前工作树**，不会自动加载稳定 ZIP：

```powershell
& $PY -m swarm_rescue.launcher --config reports/fixed_zones_work/plan_Map08.yml --bombs-file reports/fixed_zones_work/bombs_Map08.json --competition
```

该工具每次生成 `reports/fixed_zones_work/plan_*.yml` 和 `bombs_*.json`，所以**不要并行跑会写同一地图的仿真**。务必用同地图、同固定坐标、同引擎种子、同开关做 A/B；检查 JSON 中的 `disposed`、`official_total`、`elapsed_timestep`、`mean_health_percent`、`ok` 与 `official_disposal_events`（需打开事件审计）。`summary.json` 的 `source_sha256` 只标识该次工作树源码，不能当作 ZIP 哈希。

### 真正按稳定 ZIP 内源码重跑

当前 `main` 的 `tools/` **没有** `test_packaged_fixed_suite.py`。它在 `v7/gateway-egotrail` 分支；用该分支的独立 worktree 执行其 `--zip`、`--label`、`--engine-seed` 参数。该工具把提交 ZIP 安装在临时 `solutions/` 中调用官方 Launcher，避免当前工作树的不同源码污染成绩。例如在已经检出 V7 的独立目录运行：

```powershell
$PY = 'E:\envs\swarm_rescue\python.exe'
$env:PYTHONPATH = 'src'
& $PY tools/test_packaged_fixed_suite.py --zip 'E:\code\final_fusai\swarm-eod-main\artifacts\latest_upload\team006_final.zip' --label handoff_packaged_e861 --engine-seed 861
```

### V9 仅用于隔离实验

在独立的 `v9/independent-controller` 工作树中：

```powershell
$PY = 'E:\envs\swarm_rescue\python.exe'
$env:PYTHONPATH = 'src'
& $PY -m pytest -q tests/test_v9_core.py tests/test_v9_adapter_mock.py
$env:V9_CONTROLLER = '1'
$env:V6_EVENT_AUDIT = '1'
& $PY tools/run_fixed_suite.py --label handoff_v9_map01_e861 --engine-seed 861 --maps Map01
Remove-Item Env:V9_CONTROLLER
```

正式比较时先对照 `V9_CONTROLLER=0` 的旧策略，再打开 V9。不要把 V9 当前差成绩的分支直接打包提交。若以后新策略通过十图、多种子、载弹与提交格式验证，再在隔离目录生成新 ZIP，并确认 ZIP 根目录含 `my_drone_eval.py`、`team_info.yml`、`req.txt` 和策略包；再次执行 `scripts/check_submission.py <zip>`，记录 SHA256，然后才讨论替换稳定 ZIP。

## 8. 下一位开发者应先做什么

1. 固定一个可复现基线：优先按稳定 ZIP 重跑目标地图，而不是默认假设当前 `main/src` 与 ZIP 等价。
2. 对 Map06/08，用独立赛后真值审计定位未处置炸弹的**合法可观察站位**与实际 35 射线覆盖、路径是否连通及无人机是否真的到达；真值绝不进入比赛控制。
3. 对 Map07t4，区分地图“中心到达”和中心语义/炸弹物理接触；对 Map08/09，逐弹记录发现→认领→抓取→载弹返航→官方处置的中断点，并测量无 GPS、无通信区的影响。
4. 每次只改一个通用机制，保留开关和原始失败 JSON；相同种子 A/B 后回归所有已满拆地图。不能为了单图收益牺牲原 10/10，也不能用抓取数、探索面积或单元测试代替官方处置。
5. 全图均满拆后，再优化官方时间分与返航健康；Map01 的分数门槛例外仅是“不需 92”，**不是允许漏弹**。
