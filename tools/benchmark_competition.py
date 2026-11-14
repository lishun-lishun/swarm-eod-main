"""Headless official-engine benchmark for the fused 10-drone strategy."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))


def zones_for(map_name):
    if map_name=="Map06": return ["NO_COM_ZONE"]
    if map_name=="Map08": return ["NO_GPS_ZONE"]
    if map_name=="Map09": return ["NO_COM_ZONE","NO_GPS_ZONE"]
    return []


def map_class(map_name):
    suffix="07t4" if map_name=="Map07t4" else map_name[-2:]
    module=importlib.import_module(f"swarm_rescue.maps.map_{suffix}")
    return getattr(module,map_name)


def generate_layout(map_name,seed,count=10):
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.bomb import Bomb
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    zones=zones_for(map_name); zone_tuple=tuple(ZoneType[name] for name in zones)
    mission_map=map_class(map_name)(drone_type=DroneMotionless,number_drones=10,zones_config=zone_tuple)
    rng=random.Random(seed); bombs=[]
    try:
        width,height=mission_map.size_area
        for _ in range(count):
            probe=Bomb(disposal_center=mission_map.disposal_center)
            for _attempt in range(50000):
                x=rng.uniform(-width/2+32,width/2-32);y=rng.uniform(-height/2+32,height/2-32)
                pose=((x,y),0.0)
                if mission_map.playground.overlaps_collision_type(probe,pose,CollisionTypes.WALL):continue
                if mission_map.playground.overlaps_collision_type(probe,pose,CollisionTypes.DISPOSAL_CENTER):continue
                if any(math.hypot(x-b["x"],y-b["y"])<24 for b in bombs):continue
                bombs.append({"x":x,"y":y,"theta":0.0});break
            else:raise RuntimeError(f"could not place bomb {len(bombs)+1}")
    finally:mission_map.playground.cleanup()
    return {"map_name":map_name,"zones_config":zones,"bombs":bombs}


def write_inputs(map_name,seed):
    work=ROOT/"reports"/"competition_work";work.mkdir(parents=True,exist_ok=True)
    plan=work/f"plan_{map_name}_{seed}.yml";bombs=work/f"bombs_{map_name}_{seed}.json"
    zones=zones_for(map_name); zone_yaml="["+", ".join(zones)+"]" if zones else "[]"
    plan.write_text("\n".join(["stat_saving_enabled: false","state_recording_enabled: false",
        "video_capture_enabled: false","exploration_threshold_for_time_bonus: 90","evaluation_plan:",
        f"  - map_name: {map_name}","    team_mode: rescue","    nb_rounds: 1","    config_weight: 1",
        "    number_drones: 10",f"    zones_config: {zone_yaml}",""]),encoding="utf-8")
    if not bombs.exists():bombs.write_text(json.dumps(generate_layout(map_name,seed),indent=2),encoding="utf-8")
    return plan,bombs


def parse(text):
    rescued=re.search(r"rescued nb:\s*(\d+)/(\d+)",text)
    score=re.search(r"total\s+([0-9.]+)/100",text)
    health=re.search(r"mean percentage of drones health\s*:\s*([0-9.]+)",text)
    full=re.search(r"time to dispose all:\s*([0-9]+)",text)
    elapsed=re.search(r"elapse timestep:\s*([0-9]+)/(\d+)",text)
    wall=re.search(r"walltime elapsed:\s*([0-9.]+)s/([0-9.]+)s",text)
    return {"disposed":None if not rescued else int(rescued.group(1)),
        "total_bombs":None if not rescued else int(rescued.group(2)),
        "official_total":None if not score else float(score.group(1)),
        "mean_health_percent":None if not health else float(health.group(1)),
        "time_to_dispose_all_step":None if not full else int(full.group(1)),
        "elapsed_timestep":None if not elapsed else int(elapsed.group(1)),
        "max_timestep":None if not elapsed else int(elapsed.group(2)),
        "walltime_seconds":None if not wall else float(wall.group(1)),"raw_output":text}


def run(map_name,seed,engine_seed=861):
    from swarm_rescue.launcher import Launcher
    from swarm_rescue.solutions.blue_pid_explore.pose_sict_drone import (
        controller_state_snapshot, diagnostics_snapshot,
    )
    random.seed(engine_seed);np.random.seed(engine_seed);plan,bombs=write_inputs(map_name,seed)
    # ClosedPlayground intentionally asks default_rng(None).  Override only
    # during this benchmark process so sensor/actuator noise is reproducible;
    # controller code never receives this seed or the resulting true state.
    original_default_rng = np.random.default_rng
    np.random.default_rng = lambda value=None: original_default_rng(
        engine_seed if value is None else value
    )
    stream=io.StringIO()
    try:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            launcher=Launcher(config_path=str(plan),default_bombs_file=str(bombs))
            ok=launcher.go(hide_solution_output=False,headless=True,competition_mode=True)
    finally:
        np.random.default_rng = original_default_rng
    result=parse(stream.getvalue());result.update(map=map_name,seed=seed,
        engine_seed=engine_seed,ok=bool(ok),plan=str(plan),bomb_file=str(bombs))
    result["controller_diagnostics"] = diagnostics_snapshot()
    result["controller_states"] = controller_state_snapshot()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--map",default="Map01")
    parser.add_argument("--seed",type=int,default=1);parser.add_argument("--out",type=Path)
    parser.add_argument("--engine-seed",type=int,default=861)
    args=parser.parse_args();result=run(args.map,args.seed,args.engine_seed);payload=json.dumps(result,indent=2)
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(payload,encoding="utf-8")
    else:print(payload)


if __name__=="__main__":main()
