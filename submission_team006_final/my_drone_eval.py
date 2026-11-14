"""Official submission entry for the fused blue strategy."""

from swarm_rescue.solutions.blue_pid_explore.final_competition_drone import FinalCompetitionDrone
from swarm_rescue.solutions.my_drone_place_example import MyDronePlaceExample


def drone_class_for_mode(mode: str):
    return MyDronePlaceExample if str(mode).lower() == "place" else FinalCompetitionDrone


MyDroneEval = FinalCompetitionDrone
