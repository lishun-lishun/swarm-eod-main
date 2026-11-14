"""Composition of LiDAR geometry and joint nonlinear predictive safety."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .lidar_geometry import LidarBoundary, LidarGeometryEstimator
from .multi_constraint_braking import JointSafetyOutput, MultiConstraintBrakingSolver


@dataclass(frozen=True)
class LidarSafetyOutput:
    command_world: np.ndarray
    boundaries: tuple[LidarBoundary, ...]
    joint: JointSafetyOutput
    replan_required: bool


class LidarPredictiveSafetyLayer:
    def __init__(self, geometry: LidarGeometryEstimator | None = None,
                 solver: MultiConstraintBrakingSolver | None = None):
        self.geometry = geometry or LidarGeometryEstimator()
        self.solver = solver or MultiConstraintBrakingSolver()

    def reset(self) -> None:
        self.geometry.reset()
        self.solver.reset()

    def compute(
        self,
        *,
        lidar_distances,
        lidar_angles,
        estimated_position: Sequence[float],
        estimated_heading: float,
        estimated_velocity: Sequence[float],
        nominal_thrust: Sequence[float],
        footprint_vertices: Sequence[Sequence[float]],
        velocity_sigma: float = 0.0,
        position_sigma: float = 0.0,
        dynamic_boundaries: Sequence[LidarBoundary] = (),
    ) -> LidarSafetyOutput:
        boundaries = self.geometry.estimate(
            lidar_distances, lidar_angles, estimated_position, estimated_heading,
            footprint_vertices,
        )
        # Similar normals are nested half-planes: only the closest one can be
        # active.  Farther constraints are rigorously dominated.  A finite
        # relevance horizon follows the identified maximum stopping distance;
        # distant walls cannot constrain this frame and are reconsidered on the
        # next scan.  This keeps 10-drone control within the wall-clock budget.
        relevance = 28.0 + self.solver._model.stopping_distance(
            float(np.linalg.norm(estimated_velocity)) + max(0.0, velocity_sigma)
        )
        selected = []
        for boundary in sorted(boundaries, key=lambda item: item.clearance):
            if boundary.clearance > relevance:
                continue
            if any(float(np.dot(boundary.normal, kept.normal)) > math.cos(math.radians(18))
                   for kept in selected):
                continue
            selected.append(boundary)
            if len(selected) >= 6:
                break
        boundaries = selected
        boundaries.extend(dynamic_boundaries)
        joint = self.solver.compute(
            velocity=estimated_velocity,
            nominal_thrust=nominal_thrust,
            boundaries=boundaries,
            velocity_sigma=velocity_sigma,
            position_sigma=position_sigma,
        )
        return LidarSafetyOutput(joint.safe_thrust.copy(), tuple(boundaries), joint, joint.blocked)

