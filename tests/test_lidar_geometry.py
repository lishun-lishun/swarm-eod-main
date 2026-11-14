import math

import numpy as np

from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import (
    LidarGeometryEstimator, footprint_support,
)


SQUARE = np.array([[-12.0, -8.0], [12.0, -8.0], [12.0, 8.0], [-12.0, 8.0]])


def wall_scan(x_wall=60.0, rays=181):
    angles = np.linspace(-math.pi, math.pi, rays)
    distances = np.full(rays, np.nan)
    mask = np.cos(angles) > 0.4
    distances[mask] = x_wall / np.cos(angles[mask])
    distances[distances >= 230.0] = np.nan
    return distances, angles


def test_lidar_fits_wall_normal_not_individual_ray_direction():
    distances, angles = wall_scan()
    estimator = LidarGeometryEstimator()
    boundaries = estimator.estimate(distances, angles, (0, 0), 0.0, SQUARE)
    assert boundaries
    wall = min(boundaries, key=lambda item: item.clearance)
    assert wall.normal == pytest.approx([1.0, 0.0], abs=0.03)
    assert wall.surface_distance == pytest.approx(60.0, abs=0.2)
    assert wall.clearance == pytest.approx(48.0, abs=0.3)
    assert wall.point_count > 20


def test_no_return_nan_is_not_an_obstacle():
    angles = np.linspace(-math.pi, math.pi, 181)
    estimator = LidarGeometryEstimator()
    assert estimator.estimate(np.full(181, np.nan), angles, (0, 0), 0.0, SQUARE) == []


def test_actual_polygon_support_rotates_with_heading():
    assert footprint_support(SQUARE, [1, 0], 0.0) == pytest.approx(12.0)
    assert footprint_support(SQUARE, [1, 0], math.pi / 2) == pytest.approx(8.0)


def test_temporal_filter_tightens_immediately_and_releases_slowly():
    estimator = LidarGeometryEstimator(release_alpha=0.25)
    angles = wall_scan()[1]
    first = estimator.estimate(wall_scan(60)[0], angles, (0, 0), 0.0, SQUARE)[0]
    tight = estimator.estimate(wall_scan(50)[0], angles, (0, 0), 0.0, SQUARE)[0]
    release = estimator.estimate(wall_scan(70)[0], angles, (0, 0), 0.0, SQUARE)[0]
    assert tight.clearance < first.clearance
    assert tight.clearance < release.clearance < 70.0 - 12.0


import pytest

