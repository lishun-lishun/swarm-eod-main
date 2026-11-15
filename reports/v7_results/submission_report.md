# 官方提交 ZIP 与固定炸弹位置实测

## 交付包

- 上传文件：`artifacts/v7_delivery/team006_final.zip`。
- 来源：历史稳定比赛包的逐字节副本；V7 主动实验代码没有放入这个 ZIP。
- SHA256：`02193D3B368AE2471FD163BDB090428FAEB02664DAFB3AD729F4141BD21FD6E3`。
- 检查：`E:\envs\swarm_rescue\python.exe scripts/check_submission.py artifacts/v7_delivery/team006_final.zip`，结果 `Submission check passed.`；`my_drone_eval.py`、`team_info.yml`、`req.txt` 均在 ZIP 根目录。

## 按 ZIP 内真实源码隔离运行的固定布局结果

使用 `tools/test_packaged_fixed_suite.py` 将 ZIP 安装到隔离的 `solutions/`，再通过原项目官方 Launcher/Pymunk、10 架无人机、每图 `config/zones_location.txt` 的 10 枚固定炸弹、无 GPS/无通信区域、引擎种子 861 运行。结果不是 V7 工作区代码的推算。逐图原始结果与官方分数在 `reports/fixed_suites/v7_packaged_all_e861/`；汇总 JSON 中记录了实际 ZIP 的 SHA256。

| 地图 | 官方处置 | 官方分 | 健康% | 完成/终止步数 |
|---|---:|---:|---:|---:|
| Map01 | 10/10 | 83.8 | 93.2 | 1639 |
| Map02 | 10/10 | 96.7 | 91.6 | 1781 |
| Map03 | 10/10 | 99.1 | 95.6 | 984 |
| Map04 | 10/10 | 97.7 | 98.4 | 1855 |
| Map05 | 10/10 | 98.5 | 92.4 | 1949 |
| Map06 | 7/10 | 58.1 | 80.4 | 5400 |
| Map07 | 10/10 | 89.0 | 91.2 | 3367 |
| Map07t4 | 8/10 | 64.1 | 80.4 | 5400 |
| Map08 | 8/10 | 65.0 | 84.8 | 5400 |
| Map09 | 10/10 | 85.8 | 86.0 | 3770 |
| **合计/平均** | **93/100，7 张满拆** | **83.78** | **89.4** | — |

当前提交包满足官方格式与已验证的稳定基线，但**未达到所有地图 10/10 或平均 92 分**。Map01 可满拆，但不要求 92 分；其余地图中 Map06、Map07t4、Map08 仍未满拆，Map07/09 分数也未达到 92。V7 实验分支保存进一步开发所需的代码与失败证据；稳定上传包未被失败候选覆盖。
