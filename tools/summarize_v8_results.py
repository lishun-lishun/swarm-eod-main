"""Summarize official V8 Pymunk ablations and offline per-bomb lifecycle."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from summarize_v6_results import lifecycle

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v8_results"
MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map06",
        "Map07", "Map07t4", "Map08", "Map09")


def read(label, name, seed):
    path = SUITES / label / f"{name}_e{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def row(label, name, seed):
    baseline = read(f"v7_default_all_e{seed}", name, seed)
    if baseline is None:
        baseline = read(f"v6_default_all_e{seed}", name, seed)
    current = read(label, name, seed)
    data = {"label": label, "map": name, "engine_seed": seed,
            "baseline": None, "candidate": None}
    for field, source in (("baseline", baseline), ("candidate", current)):
        if source is None:
            continue
        events = [e for entries in source.get("controller_task_events", {}).values()
                  for e in entries]
        counts = Counter(e.get("kind") for e in events)
        data[field] = {
            "official_disposed": source.get("disposed"),
            "official_score": source.get("official_total"),
            "health_percent": source.get("mean_health_percent"),
            "elapsed_steps": source.get("elapsed_timestep"),
            "control_p95_ms": source.get("controller_diagnostics", {}).get(
                "control_p95_ms"),
            "macro_planned": counts["v8_macro_planned"],
            "macro_reached": sum(e.get("reached", False) for e in events
                                 if e.get("kind") == "v8_macro_end"),
            "ego_candidates": counts["v8_route_candidate"],
            "ego_actions": counts["v8_route_action"],
        }
    if baseline and current:
        data["delta_official_disposed"] = (
            current["disposed"] - baseline["disposed"])
        data["delta_official_score"] = round(
            current["official_total"] - baseline["official_total"], 3)
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("v8_macro_shadow_e861", "Map06", 861),
        ("v8_macro_shadow_e861", "Map08", 861),
        ("v8_macro_active_e861", "Map06", 861),
        ("v8_macro_active_e861", "Map08", 861),
        ("v8_ego_shadow_09_e862", "Map09", 862),
    ]
    default = [(f"v8_default_all_e{seed}", name, seed)
               for seed in (861, 862, 863) for name in MAPS]
    report = {
        "engine": "official Launcher/Pymunk, 10 drones, 10 fixed bombs",
        "switches_default": {
            "V8_MACRO_SHADOW": "0", "V8_MACRO_ACTIVE": "0",
            "V8_EGO_SHADOW": "0", "V8_EGO_ACTIVE": "0"},
        "experiments": [row(*x) for x in experiments],
        "default_full_suites": [row(*x) for x in default],
        "decision": {
            "macro_active": "REJECT: Map08 seed861 8/10 to 6/10",
            "ego_active": "REJECT_UNTESTED: Map09 drone6 has no valid GPS route anchor",
            "stable_zip_replaced": False,
        },
    }
    (OUT / "ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = {}
    for label, name, seed in experiments:
        result = read(label, name, seed)
        if result is None:
            continue
        failures[f"{label}/{name}/e{seed}"] = {
            "per_bomb_lifecycle_offline": lifecycle(result),
            "official_disposal_events": result.get("official_disposal_events", []),
            "v8_events": {
                drone: [e for e in events if e.get("kind", "").startswith("v8_")]
                for drone, events in result.get("controller_task_events", {}).items()
                if any(e.get("kind", "").startswith("v8_") for e in events)},
            "v8_counts": {
                str(d["id"]): d.get("v8_counts")
                for d in result.get("controller_states", [])
                if d.get("v8_counts")},
        }
    (OUT / "failure_trace.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V8 默认关闭：三种子官方固定布局逐图成绩",
        "",
        "官方 Launcher/Pymunk，10 架无人机，每图 10 枚固定炸弹，含无 GPS 与无通信区。以下是 V8 四项主动/影子开关全部关闭的真实测试；原始 JSON 在 `reports/fixed_suites/v8_default_all_e861/`、`e862/`、`e863/`。",
        "",
        "| 地图 | seed861 处置/分/健康%/步数 | seed862 处置/分/健康%/步数 | seed863 处置/分/健康%/步数 |",
        "|---|---:|---:|---:|",
    ]
    for name in MAPS:
        values = []
        for seed in (861, 862, 863):
            result = read(f"v8_default_all_e{seed}", name, seed)
            values.append((f"{result['disposed']}/10 / {result['official_total']:.1f} / "
                           f"{result['mean_health_percent']:.1f} / {result['elapsed_timestep']}")
                          if result else "未完成")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    summary = []
    for seed in (861, 862, 863):
        path = SUITES / f"v8_default_all_e{seed}" / "summary.json"
        if path.exists():
            suite = json.loads(path.read_text(encoding="utf-8"))
            summary.append(f"{suite['total_disposed']}/100 / "
                           f"{suite['mean_official_score']:.2f} / "
                           f"{suite['mean_health_percent']:.1f} / "
                           f"{suite['full_maps']}张满拆")
        else:
            summary.append("未完成")
    lines.extend(["| **合计/均分** | " + " | ".join(summary) + " |", "",
                  "稳定提交 ZIP 没有被 V8 失败候选替换。V8 主动 Macro 仅在 Map06/Map08 seed861 做过配对消融，Map08 从 8/10 降至 6/10；不能把这张默认关闭表当作主动 V8 成绩。", ""])
    (OUT / "fixed_suites.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote V8 ablation and failure trace")


if __name__ == "__main__":
    main()
