"""Pair official V6 runs against stable same-layout baselines."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v6_results" / "ablation.json"
MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map06",
        "Map07", "Map07t4", "Map08", "Map09")


def raw(label: str, name: str, seed: int):
    path = SUITES / label / f"{name}_e{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def slim(data):
    if data is None:
        return None
    diag = data.get("controller_diagnostics", {})
    events = [event for rows in data.get("controller_task_events", {}).values()
              for event in rows]
    return {
        "disposed": data["disposed"], "official_score": data["official_total"],
        "health_percent": data["mean_health_percent"],
        "steps": data["elapsed_timestep"],
        "wall_seconds": data["walltime_seconds"],
        "control_p95_ms": diag.get("control_p95_ms"),
        "v6_view_planned": sum(e["kind"] == "v6_view_planned" for e in events),
        "v6_view_reached": sum(e["kind"] == "v6_view_reached" for e in events),
        "v6_verified_replan": sum(e["kind"] == "v6_connected_repair" for e in events),
        "v6_route_issues": sum(e["kind"] == "v6_carry_issue" for e in events),
        "v6_semantic_cells_added": sum(e.get("new_cells", 0) for e in events
                                      if e["kind"] == "v6_real_ray_cells"),
    }


def baseline(name: str, seed: int):
    if seed == 861:
        return raw("v5_final_default_all_e861", name, seed)
    for label in ("v5_baseline_Map09_e862", "v5_baseline_core_e862",
                  "v4_baseline_subset_e862"):
        result = raw(label, name, seed)
        if result is not None:
            return result
    return None


def lifecycle(data):
    """Offline matching; only disposal is an exact official engine event."""
    if data is None or "official_disposal_events" not in data:
        return None
    traces = [entry for drone in data.get("controller_states", [])
              for entry in drone.get("bomb_trace", [])]
    tasks = [entry for rows in data.get("controller_task_events", {}).values()
             for entry in rows]
    disposals = {entry["bomb_index"]: entry["step"]
                 for entry in data["official_disposal_events"]
                 if entry["bomb_index"] is not None}
    result = []
    for bomb in data["truth_audit_after_control"]:
        idx = bomb["index"]
        xy = bomb["position"]

        def nearest_step(kind, records, radius):
            steps = []
            key = "bomb" if kind in ("first_seen", "claim") else "pos"
            for e in records:
                if e.get("kind") != kind or key not in e:
                    continue
                distances = sorted((math.dist(e[key], other["position"]),
                                    other["index"])
                                   for other in data["truth_audit_after_control"])
                # Estimated poses drift in no-GPS zones. Attribute only when
                # this fixed bomb is the unique nearest by a useful margin.
                if (distances[0][1] == idx and distances[0][0] <= radius
                        and (len(distances) == 1
                             or distances[1][0] - distances[0][0] >= 10.0)):
                    steps.append(e["step"])
            return min(steps) if steps else None

        seen = nearest_step("first_seen", traces, 65.0)
        claim = nearest_step("claim", traces, 65.0)
        grasp = nearest_step("grasp", tasks, 65.0)
        disposed = disposals.get(idx)
        failure = (None if disposed is not None else
                   "undiscovered" if seen is None else
                   "discovered_not_grasped" if grasp is None else
                   "grasped_not_officially_disposed")
        result.append({"bomb_index": idx, "first_seen_estimate": seen,
                       "claim_estimate": claim, "grasp_estimate": grasp,
                       "official_disposal_step": disposed,
                       "failure_class": failure})
    return result


def main():
    arms = {
        "shadow": [("v6_shadow_Map06_e861", "Map06", 861),
                   ("v6_shadow_Map08_e861", "Map08", 861),
                   ("v6_shadow_Map06_e862", "Map06", 862),
                   ("v6_shadow_Map08_e862", "Map08", 862)],
        "shadow_truth": [("v6_shadow_truth_Map06_e861", "Map06", 861),
                         ("v6_shadow_truth_Map08_e861", "Map08", 861)],
        "view": [("v6_view_Map06_e861", "Map06", 861),
                 ("v6_view_Map08_e861", "Map08", 861)],
        "view_mature": [("v6_view_mature_Map08_e861", "Map08", 861)],
        "route": [("v6_route_Map09_e862", "Map09", 862),
                  ("v6_route_Map07t4_e861", "Map07t4", 861)],
        "task": [("v6_task_Map09_e862", "Map09", 862),
                 ("v6_task_Map04_e861", "Map04", 861)],
        "event_audit": [("v6_event_audit_Map04_e861", "Map04", 861)],
        "brake_shadow": [("v6_brake_shadow_Map07_e861", "Map07", 861)],
        "default_861": [("v6_default_all_e861", name, 861) for name in MAPS],
        "default_862": [("v6_default_all_e862", name, 862) for name in MAPS],
    }
    report = {"engine": "official Launcher/Pymunk", "arms": {}}
    for arm, entries in arms.items():
        rows = []
        for label, name, seed in entries:
            data = raw(label, name, seed)
            if data is None:
                continue
            control = baseline(name, seed)
            rows.append({"map": name, "seed": seed, "label": label,
                         "baseline": slim(control), "candidate": slim(data),
                         "bomb_lifecycle": lifecycle(data)})
        report["arms"][arm] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
