"""Summarize paired official V7 experiments without feeding truth to policy."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from summarize_v6_results import lifecycle

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v7_results"
MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map06",
        "Map07", "Map07t4", "Map08", "Map09")


def load(label: str, name: str, seed: int):
    path = SUITES / label / f"{name}_e{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def metrics(data):
    if data is None:
        return None
    events = [event for rows in data.get("controller_task_events", {}).values()
              for event in rows]
    counts = Counter(event.get("kind") for event in events)
    gateway_end = Counter(event.get("reason") for event in events
                          if event.get("kind") == "v7_gateway_end")
    return {
        "disposed_official": data["disposed"],
        "score_official": data["official_total"],
        "health_percent": data["mean_health_percent"],
        "elapsed_steps": data["elapsed_timestep"],
        "wall_seconds": data["walltime_seconds"],
        "control_p95_ms": data.get("controller_diagnostics", {}).get("control_p95_ms"),
        "gateway_planned": counts["v7_gateway_planned"],
        "gateway_reached": counts["v7_gateway_reached"],
        "gateway_end_reasons": dict(gateway_end),
        "probe_confirmed": counts["v7_probe_confirmed"],
        "probe_reached": counts["v7_probe_reached"],
        "ego_verified_events": counts["v7_ego_retreat_candidate"],
        "ego_action_frames": counts["v7_ego_retreat_action"],
        "real_semantic_ray_cells_added": sum(
            int(event.get("new_cells", 0)) for event in events
            if event.get("kind") == "v6_real_ray_cells"),
    }


def compare(label, name, seed):
    baseline = load(f"v6_default_all_e{seed}", name, seed)
    candidate = load(label, name, seed)
    row = {"label": label, "map": name, "engine_seed": seed,
           "baseline": metrics(baseline), "candidate": metrics(candidate)}
    if baseline and candidate:
        row["delta_disposed"] = candidate["disposed"] - baseline["disposed"]
        row["delta_score"] = round(
            candidate["official_total"] - baseline["official_total"], 3)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("v7_shadow_06_e861", "Map06", 861),
        ("v7_shadow_08_e861", "Map08", 861),
        ("v7_gateway_06_e861", "Map06", 861),
        ("v7_gateway_06_e861_fix1", "Map06", 861),
        ("v7_gateway_08_e861_fix1", "Map08", 861),
        ("v7_gateway_08_e861_idle", "Map08", 861),
        ("v7_gateway_late_e861", "Map06", 861),
        ("v7_gateway_late_e861", "Map08", 861),
        ("v7_gateway_probe_e861", "Map06", 861),
        ("v7_gateway_probe_e861", "Map08", 861),
        ("v7_ego_shadow_09_e862", "Map09", 862),
        ("v7_ego_shadow_09_e862_stall", "Map09", 862),
        ("v7_ego_d6_active_09_e862", "Map09", 862),
    ]
    experiments.extend(
        (f"v7_shadow_key_e{seed}", name, seed)
        for seed in (862, 863)
        for name in ("Map06", "Map07t4", "Map08", "Map09"))
    rows = [compare(*experiment) for experiment in experiments]
    default = [compare("v7_default_all_e861", name, 861) for name in MAPS]
    report = {
        "engine": "official Launcher/Pymunk, ten drones, fixed ten bombs, GPS and communication denial zones",
        "policy_source": "V6 stable with independently switchable V7 prototypes",
        "active_v7_defaults": False,
        "experiments": rows,
        "default_full_suite_seed861": default,
        "decision": {
            "gateway_active": "REJECT_DEFAULT: Map08 seed861 lost one official disposal",
            "ego_retreat_active": "REJECT_DEFAULT: Map09 seed862 did not increase disposal and reduced score/health",
            "probe_active": "REJECT_DEFAULT: Map08 seed861 gateway plus probe lost one official disposal",
            "yaw_sweep_active": "UNTESTED_IN_OFFICIAL_ENGINE: gateway prerequisite rejected",
            "stable_zip_replaced": False,
        },
    }
    (OUT / "ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = {}
    for label, name, seed in (
        ("v7_shadow_06_e861", "Map06", 861),
        ("v7_gateway_06_e861_fix1", "Map06", 861),
        ("v7_shadow_08_e861", "Map08", 861),
        ("v7_gateway_08_e861_fix1", "Map08", 861),
        ("v7_gateway_late_e861", "Map08", 861),
        ("v7_gateway_probe_e861", "Map06", 861),
        ("v7_gateway_probe_e861", "Map08", 861),
        ("v7_ego_shadow_09_e862_stall", "Map09", 862),
        ("v7_ego_d6_active_09_e862", "Map09", 862),
    ):
        data = load(label, name, seed)
        if data is None:
            continue
        key = f"{label}/{name}/e{seed}"
        failures[key] = {
            "per_bomb_offline_lifecycle": lifecycle(data),
            "official_disposals": data.get("official_disposal_events", []),
            "v7_events": {
                drone: [event for event in events
                        if event.get("kind", "").startswith("v7_")]
                for drone, events in data.get("controller_task_events", {}).items()
                if any(event.get("kind", "").startswith("v7_") for event in events)
            },
            "per_drone_counts": {
                str(drone["id"]): drone.get("v7_counts", {})
                for drone in data.get("controller_states", [])
                if drone.get("v7_counts")
            },
        }
    (OUT / "failure_trace.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'ablation.json'} and failure_trace.json")


if __name__ == "__main__":
    main()
