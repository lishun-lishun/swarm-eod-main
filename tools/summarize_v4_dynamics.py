"""Record calibrated empty dynamics and competition-legal cargo shadow fits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    measured = json.loads((ROOT / "reports/dynamics_identification.json").read_text(
        encoding="utf-8"))
    path = (ROOT / "reports/fixed_suites/v4_cargo_shadow_holdout_e861"
            / "Map09_e861.json")
    shadow = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    drones = ([] if shadow is None else [
        {"id": row["id"], **row["cargo_model"]}
        for row in shadow["controller_states"]
        if row.get("cargo_model") is not None
    ])
    report = {
        "empty_model_source": "reports/dynamics_identification.json",
        "empty_model_method": measured["status"],
        "empty_velocity_decay": measured["scalar_model"]["velocity_decay"],
        "empty_input_gain": measured["scalar_model"]["input_matrix"][0][0],
        "empty_validation_rmse": measured["scalar_model"]["validation_rmse"],
        "cargo_observable": "noisy odometer frame displacement (px/control frame)",
        "cargo_source": str(path.relative_to(ROOT)) if shadow else None,
        "cargo_official_disposed": shadow["disposed"] if shadow else None,
        "cargo_official_score": shadow["official_total"] if shadow else None,
        "cargo_drones": drones,
        "cargo_total_updates": sum(row["updates"] for row in drones),
        "cargo_total_holdout_pairs": sum(row["holdout_pairs"] for row in drones),
        "cargo_control_authorized": False,
        "limitation": ("The fitted observable is frame-average odometer displacement; "
                       "it is not proven to be endpoint velocity or valid during "
                       "wall contact, bomb swing, pickup, or terminal delivery."),
    }
    (ROOT / "reports/dynamics_calibration_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
