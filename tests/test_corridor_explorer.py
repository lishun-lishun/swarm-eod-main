import math

import numpy as np

from swarm_rescue.solutions.blue_pid_explore.corridor_explorer import (
    CorridorExplorer, CorridorExploreState,
)
from swarm_rescue.solutions.blue_pid_explore.oriented_footprint import OrientedFootprint


ANGLES = np.linspace(-math.pi, math.pi, 181)


def footprint():
    return OrientedFootprint([[-15, -10], [15, -10], [15, 10], [-15, 10]])


def corridor_scan(end_x=150.0, half_width=40.0):
    values = np.full_like(ANGLES, 230.0)
    for index, angle in enumerate(ANGLES):
        candidates = []
        cosine, sine = math.cos(angle), math.sin(angle)
        if cosine > 1e-6:
            candidates.append(end_x / cosine)
        if sine > 1e-6:
            candidates.append(half_width / sine)
        elif sine < -1e-6:
            candidates.append(-half_width / sine)
        if candidates:
            values[index] = min(230.0, min(v for v in candidates if v > 0.0))
    return values


def update(explorer, scan, *, step, velocity=(0.0, 0.0)):
    return explorer.update(
        step=step, position=(float(step), 0.0), velocity=velocity,
        heading=0.0, travel_heading=0.0,
        lidar_distances=scan, lidar_angles=ANGLES,
    )


def test_corridor_requires_multiple_frames_then_commits():
    explorer = CorridorExplorer(footprint(), entry_confirm_frames=3)
    scan = corridor_scan()
    assert not update(explorer, scan, step=0, velocity=(1.0, 0.0)).active
    assert not update(explorer, scan, step=1, velocity=(1.0, 0.0)).active
    output = update(explorer, scan, step=2, velocity=(1.0, 0.0))
    assert output.active
    assert output.state == CorridorExploreState.ADVANCE
    assert output.corridor_width == pytest_approx(80.0, 3.0)


def pytest_approx(value, tolerance):
    class Approx:
        def __eq__(self, other):
            return abs(float(other) - value) <= tolerance
    return Approx()


def test_end_gap_subtracts_convex_support_once_and_slows():
    explorer = CorridorExplorer(footprint())
    scan = corridor_scan(end_x=22.0)
    explorer.state = CorridorExploreState.END_APPROACH
    explorer.direction = 0.0
    explorer.started_step = 0
    output = update(explorer, scan, step=3, velocity=(0.2, 0.0))
    # Center-origin range 22 minus x support 15 => 7 px surface gap.
    assert abs(output.end_gap - 7.0) < 0.5
    assert 0.0 <= output.target_velocity[0] <= 1.35
    assert output.safe_gap >= 2.0


def test_single_noisy_ray_is_not_a_branch_but_continuous_gap_is():
    explorer = CorridorExplorer(footprint(), confirm_frames=3)
    explorer.state = CorridorExploreState.INSPECT
    explorer.direction = 0.0
    explorer.started_step = 0
    base = corridor_scan(end_x=20.0)
    noisy = base.copy()
    noisy[np.argmin(np.abs(ANGLES - math.pi / 2.0))] = 230.0
    for step in range(2):
        output = update(explorer, noisy, step=step)
        assert output.state == CorridorExploreState.INSPECT
    gap = base.copy()
    gap[(ANGLES > 0.65) & (ANGLES < 1.05)] = 230.0
    for step in range(2, 5):
        output = update(explorer, gap, step=step)
    assert output.state == CorridorExploreState.BRANCH
    assert output.branch_angle is not None
    assert explorer.stats["branches_found"] == 1


def test_safe_gap_grows_with_toward_wall_speed():
    explorer = CorridorExplorer(footprint())
    assert explorer._safe_gap(4.0) > explorer._safe_gap(0.0) > 2.0

