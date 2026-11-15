"""Official Launcher/Pymunk one-drone, one-bomb contact experiment.

The test fixture chooses the bomb location. V9 receives only its normal
sensor observations; disposal and cargo truth are recorded after control.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_competition import parse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="Map01")
    parser.add_argument("--engine-seed", type=int, default=861)
    parser.add_argument("--bomb-x", type=float, default=-455.)
    parser.add_argument("--bomb-y", type=float, default=95.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from swarm_rescue.launcher import Launcher
    from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
    from swarm_rescue.solutions.v9_rescue.diagnostics import snapshot

    work = ROOT / "reports" / "v9_results" / "single_bomb_work"
    work.mkdir(parents=True, exist_ok=True)
    plan = work / "plan.yml"
    bombs = work / "bombs.json"
    plan.write_text("\n".join((
        "stat_saving_enabled: false", "state_recording_enabled: false",
        "video_capture_enabled: false", "exploration_threshold_for_time_bonus: 90",
        "evaluation_plan:", f"  - map_name: {args.map}",
        "    team_mode: rescue", "    nb_rounds: 1", "    config_weight: 1",
        "    number_drones: 1", "    zones_config: [NO_COM_ZONE, NO_GPS_ZONE]", "")),
        encoding="utf-8")
    bombs.write_text(json.dumps({"map_name": args.map,
        "zones_config": ["NO_COM_ZONE", "NO_GPS_ZONE"],
        "bombs": [{"x": args.bomb_x, "y": args.bomb_y, "theta": 0.}]}),
        encoding="utf-8")
    random.seed(args.engine_seed)
    np.random.seed(args.engine_seed)
    original_rng = np.random.default_rng
    original_activate = DisposalCenter.activate
    events = []

    def observed_activate(center, bomb):
        if not bomb.disposed:
            events.append({"step": int(center._playground.timestep),
                           "grasped": bool(bomb.grasped_by)})
        return original_activate(center, bomb)

    np.random.default_rng = lambda value=None: original_rng(
        args.engine_seed if value is None else value)
    DisposalCenter.activate = observed_activate
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            launcher = Launcher(config_path=str(plan), default_bombs_file=str(bombs))
            ok = launcher.go(hide_solution_output=False, headless=True,
                             competition_mode=True)
    finally:
        DisposalCenter.activate = original_activate
        np.random.default_rng = original_rng
    result = parse(stream.getvalue())
    result.update(ok=bool(ok), map=args.map, engine_seed=args.engine_seed,
                  bomb_xy=[args.bomb_x,args.bomb_y],
                  official_disposal_events=events, v9_legal_agents=snapshot())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"one bomb: {result['disposed']}/{result['total_bombs']} "
          f"at step {events[0]['step'] if events else None}")
    if not ok:
        raise RuntimeError("Official Launcher failed; inspect raw_output")


if __name__ == "__main__":
    main()
