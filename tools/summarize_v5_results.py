"""Pair raw official Pymunk V5 runs with same-layout, same-seed baselines."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v5_results"


def load(label: str, name: str, seed: int) -> dict | None:
    path = SUITES / label / f"{name}_e{seed}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "disposed": data["disposed"],
        "score": data["official_total"],
        "steps": data["elapsed_timestep"],
        "health": data["mean_health_percent"],
        "grasps": data.get("controller_diagnostics", {}).get("grasp_successes"),
        "stuck": data.get("controller_diagnostics", {}).get("stuck_events"),
        "events": data.get("controller_diagnostics", {}).get("event_counts", {}),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def source_sha(label: str) -> str | None:
    path = SUITES / label / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("source_sha256")


def baseline(name: str, seed: int) -> dict | None:
    if seed == 861:
        return (load("v5_final_default_all_e861", name, seed)
                or load("baseline_20261114_e861", name, seed))
    return (load("v5_baseline_Map09_e862", name, seed)
            or load("v5_baseline_core_e862", name, seed)
            or load("v4_baseline_subset_e862", name, seed))


def main() -> None:
    arms = {
        "watchdog": [
            ("v5_watchdog_Map09_e862", 862, ("Map09",)),
            ("v5_watchdog_Map08_e861", 861, ("Map08",)),
        ],
        "deadline": [
            ("v5_deadline_Map09_e862", 862, ("Map09",)),
            ("v5_deadline_core_e862", 862, ("Map06", "Map08")),
            ("v5_deadline_core_e861", 861,
             ("Map07t4", "Map06", "Map08", "Map04", "Map07", "Map09")),
        ],
        "recover_first": [
            ("v5_recover_Map09_e862", 862, ("Map09",)),
            ("v5_recover_core_e861", 861,
             ("Map07t4", "Map06", "Map08", "Map04", "Map07", "Map09")),
        ],
        "recover_empty_only": [
            ("v5_recover_empty_Map09_e862", 862, ("Map09",)),
            ("v5_recover_empty_core_e862", 862, ("Map06", "Map08")),
            ("v5_recover_empty_core_e861", 861,
             ("Map06", "Map08", "Map07t4")),
        ],
        "recover_full_hull": [
            ("v5_recover_hull_e862", 862, ("Map09",)),
        ],
        "recover_endgame": [
            ("v5_recover_endgame_Map09_e862", 862, ("Map09",)),
            ("v5_recover_endgame_Map06_e861", 861, ("Map06",)),
            ("v5_recover_endgame_Map08_e861", 861, ("Map08",)),
        ],
        "blend_alpha_0_1": [
            ("v5_blend_a01_e861", 861, ("Map04", "Map07", "Map09")),
        ],
        "final_default": [
            ("v5_final_default_all_e861", 861,
             ("Map01", "Map02", "Map03", "Map04", "Map05",
              "Map06", "Map07", "Map07t4", "Map08", "Map09")),
        ],
    }
    out = {"engine": "official Launcher/Pymunk", "layout": "config/zones_location.txt",
           "arms": {}}
    for arm, suites in arms.items():
        rows = []
        for label, seed, maps in suites:
            for name in maps:
                candidate = load(label, name, seed)
                if candidate is None:
                    continue
                control = baseline(name, seed)
                rows.append({"map": name, "seed": seed,
                             "baseline": control, "candidate": candidate,
                             "source_sha256": source_sha(label)})
        regressions = [f'{row["map"]}:{row["seed"]}' for row in rows
                       if row["baseline"] is not None
                       and row["candidate"]["disposed"] < row["baseline"]["disposed"]]
        out["arms"][arm] = {"paired": rows, "disposal_regressions": regressions,
                            "accepted": False if regressions else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
