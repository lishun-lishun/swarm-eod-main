from pathlib import Path

from swarm_rescue.maps.map_01 import Map01
from swarm_rescue.maps.map_04 import Map04
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.reporting.evaluation import EvalPlan


def test_eval_plan_reads_number_drones_defaults_and_overrides(tmp_path: Path):
    config = tmp_path / "plan.yml"
    config.write_text(
        "\n".join(
            [
                "team_mode: rescue",
                "number_drones: 7",
                "evaluation_plan:",
                "  - map_name: Map01",
                "  - map_name: Map04",
                "    number_drones: 3",
            ]
        ),
        encoding="utf-8",
    )

    eval_plan = EvalPlan()
    assert eval_plan.from_yaml(str(config)) is True
    assert len(eval_plan.list_eval_config) == 2
    assert eval_plan.list_eval_config[0].number_drones == 7
    assert eval_plan.list_eval_config[1].number_drones == 3


def test_map_04_uses_number_drones_override():
    default_map = Map04(drone_type=DroneMotionless)
    assert default_map.number_drones == 10
    assert len(default_map.drones) == 10

    custom_map = Map04(drone_type=DroneMotionless, number_drones=4)
    assert custom_map.number_drones == 4
    assert len(custom_map.drones) == 4


def test_map_01_respects_number_drones_override():
    default_map = Map01(drone_type=DroneMotionless)
    assert default_map.number_drones == 10
    assert len(default_map.drones) == 10

    custom_map = Map01(drone_type=DroneMotionless, number_drones=6)
    assert custom_map.number_drones == 6
    assert len(custom_map.drones) == 6


def test_eval_plan_reads_time_best_ratio(tmp_path: Path):
    """The blue time curve is a top-level YAML key, and 0 is a real value."""
    config = tmp_path / "plan.yml"
    config.write_text(
        "\n".join(
            [
                "team_mode: rescue",
                "time_best_ratio: 0",
                "evaluation_plan:",
                "  - map_name: Map01",
            ]
        ),
        encoding="utf-8",
    )

    eval_plan = EvalPlan()
    assert eval_plan.time_best_ratio == 0.5
    assert eval_plan.from_yaml(str(config)) is True
    assert eval_plan.time_best_ratio == 0.0


def test_eval_plan_rejects_time_best_ratio_out_of_range(tmp_path: Path):
    config = tmp_path / "plan.yml"
    config.write_text(
        "\n".join(
            [
                "team_mode: rescue",
                "time_best_ratio: 1.5",
                "evaluation_plan:",
                "  - map_name: Map01",
            ]
        ),
        encoding="utf-8",
    )

    assert EvalPlan().from_yaml(str(config)) is False
