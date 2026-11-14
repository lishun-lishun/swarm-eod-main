import math

import numpy as np

from swarm_rescue.solutions.blue_pid_explore.corridor_controller import CorridorController, CorridorState
from swarm_rescue.solutions.blue_pid_explore.oriented_footprint import OrientedFootprint, Pose2
from swarm_rescue.solutions.blue_pid_explore.se2_local_planner import SE2LocalPlanner


RECTANGLE = np.array([[-8, -14], [8, -14], [8, 14], [-8, 14]], dtype=float)


def test_projection_width_and_best_heading():
    footprint = OrientedFootprint(RECTANGLE)
    assert footprint.projection_width(0.0, [0, 1]) == pytest.approx(28.0)
    heading, width = footprint.best_corridor_heading([0, 1], bins=32)
    assert width == pytest.approx(16.0, abs=0.1)
    assert abs(abs(heading) - math.pi / 2) < 0.15


def test_swept_rotation_detects_intermediate_collision():
    footprint = OrientedFootprint(RECTANGLE)
    grid = np.zeros((30, 30), dtype=bool)
    grid[16, 17] = True
    origin = np.array([-60.0, -60.0])
    start = Pose2(0, 0, 0); end = Pose2(0, 0, math.pi / 2)
    assert footprint.swept_collides(start, end, grid, origin, 4.0)


def test_se2_planner_rotates_to_cross_narrow_corridor():
    footprint = OrientedFootprint(RECTANGLE, margin=1.0)
    grid = np.zeros((35, 45), dtype=bool)
    # Vertical barrier with a 6-cell (24 px) horizontal passage.  The 28 px
    # body cannot pass at theta=0, but its 16 px rotated width can.
    grid[:, 21:24] = True
    grid[15:21, 21:24] = False
    planner = SE2LocalPlanner(footprint, heading_bins=16, max_expansions=50000)
    result = planner.plan(Pose2(-55, 10, 0), Pose2(55, 10, 0), grid,
                          origin=(-90, -70), resolution=4.0)
    assert result.success, result.reason
    assert any(abs(abs(p.theta) - math.pi / 2) < 0.3 for p in result.poses)


def test_corridor_rejects_cargo_that_does_not_fit():
    controller = CorridorController(OrientedFootprint(RECTANGLE))
    command = controller.compute(
        position=[0, 0], velocity=[0, 0], heading=0,
        entrance=[0, 0], exit_point=[100, 0], centerline=[1, 0],
        corridor_width=20, safety_margin=2, carrying=True, cargo_extra_width=5,
    )
    assert command.state == CorridorState.BLOCKED
    assert command.replan_required


import pytest

