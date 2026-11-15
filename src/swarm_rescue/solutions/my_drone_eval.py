"""Official submission entry with an opt-in independent V9 controller."""

import os

from swarm_rescue.solutions.blue_pid_explore.final_competition_drone import FinalCompetitionDrone
from swarm_rescue.solutions.my_drone_place_example import MyDronePlaceExample
from swarm_rescue.solutions.v9_rescue.adapter import V9RescueDrone


_USE_V9 = os.environ.get("V9_CONTROLLER", "0") == "1"


def drone_class_for_mode(mode: str):
    if str(mode).lower() == "place":
        return MyDronePlaceExample
    return V9RescueDrone if _USE_V9 else FinalCompetitionDrone


MyDroneEval = V9RescueDrone if _USE_V9 else FinalCompetitionDrone
