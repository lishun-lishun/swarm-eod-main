"""Short official-map cProfile run; evaluator may inspect truth, controllers may not."""
from __future__ import annotations
import cProfile
import json
from pathlib import Path
import pstats
import random
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from swarm_rescue.maps.map_01 import Map01
from swarm_rescue.solutions.blue_pid_explore.competition_drone import BluePidExploreDrone

random.seed(1);np.random.seed(1)
mission=Map01(drone_type=BluePidExploreDrone,number_drones=10)
layout=json.loads((ROOT/"reports"/"competition_work"/"bombs_Map01_1.json").read_text())
mission.spawn_bombs(layout["bombs"])
for drone in mission.drones:drone.set_competition_mode(True)
mission.playground._compute_observations()
profile=cProfile.Profile();profile.enable();timings=[]
try:
    for step in range(3):
        started=time.perf_counter()
        commands={drone:drone.control() for drone in mission.drones}
        control_done=time.perf_counter()
        mission.playground.step(all_commands=commands)
        timings.append({"step":step,"control_seconds":control_done-started,
                        "physics_seconds":time.perf_counter()-control_done})
finally:
    profile.disable();mission.playground.cleanup()
print(json.dumps(timings,indent=2))
pstats.Stats(profile).sort_stats("cumulative").print_stats(35)
