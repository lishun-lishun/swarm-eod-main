"""Narrow-passage state machine layered above the existing flight controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .oriented_footprint import OrientedFootprint, Pose2, angle_delta


class CorridorState(str, Enum):
    APPROACH_ENTRANCE = "APPROACH_ENTRANCE"
    ALIGN = "ALIGN"
    ENTER = "ENTER"
    TRAVERSE = "TRAVERSE"
    EXIT = "EXIT"
    RECOVER = "RECOVER"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class CorridorCommand:
    state: CorridorState
    target_position: np.ndarray
    target_velocity: np.ndarray
    target_heading: float
    rotation_command: float
    replan_required: bool


class CorridorController:
    def __init__(self, footprint: OrientedFootprint, *, position_tolerance: float = 4.0,
                 speed_tolerance: float = 0.25, heading_tolerance_deg: float = 4.0):
        self.footprint = footprint
        self.position_tolerance = float(position_tolerance)
        self.speed_tolerance = float(speed_tolerance)
        self.heading_tolerance = math.radians(heading_tolerance_deg)
        self.reset()

    def reset(self) -> None:
        self.state = CorridorState.APPROACH_ENTRANCE
        self._stable = 0

    def _hold(self, position, heading, target, target_heading, state):
        error = np.asarray(target) - np.asarray(position)
        velocity = np.clip(0.12 * error, -1.2, 1.2)
        rotation = float(np.clip(1.4 * angle_delta(target_heading, heading), -1.0, 1.0))
        return CorridorCommand(state, np.asarray(target, dtype=float), velocity,
                               float(target_heading), rotation, False)

    def compute(
        self, *, position: Sequence[float], velocity: Sequence[float], heading: float,
        entrance: Sequence[float], exit_point: Sequence[float], centerline: Sequence[float],
        corridor_width: float, safety_margin: float, carrying: bool = False,
        cargo_extra_width: float = 0.0,
    ) -> CorridorCommand:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        entrance, exit_point = np.asarray(entrance, dtype=float), np.asarray(exit_point, dtype=float)
        axis = np.asarray(centerline, dtype=float)
        axis /= max(1e-12, np.linalg.norm(axis))
        normal = np.array([-axis[1], axis[0]])
        desired_heading, width = self.footprint.best_corridor_heading(normal)
        required = width + 2.0 * safety_margin + (cargo_extra_width if carrying else 0.0)
        if required >= corridor_width:
            self.state = CorridorState.BLOCKED
            return CorridorCommand(self.state, position.copy(), np.zeros(2), heading, 0.0, True)
        if self.state == CorridorState.APPROACH_ENTRANCE:
            command = self._hold(position, heading, entrance, heading, self.state)
            if np.linalg.norm(position - entrance) <= self.position_tolerance and np.linalg.norm(velocity) <= self.speed_tolerance:
                self.state = CorridorState.ALIGN
            return command
        if self.state == CorridorState.ALIGN:
            command = self._hold(position, heading, entrance, desired_heading, self.state)
            aligned = abs(angle_delta(desired_heading, heading)) <= self.heading_tolerance
            stopped = np.linalg.norm(velocity) <= self.speed_tolerance
            self._stable = self._stable + 1 if aligned and stopped else 0
            if self._stable >= 5:
                self.state = CorridorState.ENTER; self._stable = 0
            return command
        progress = float((position - entrance) @ axis)
        length = float((exit_point - entrance) @ axis)
        lateral_error = float((position - entrance) @ normal)
        speed = 2.0 if carrying else 3.0
        target_velocity = speed * axis - 0.18 * lateral_error * normal
        rotation = float(np.clip(1.6 * angle_delta(desired_heading, heading), -0.45, 0.45))
        if self.state == CorridorState.ENTER and progress > 8.0:
            self.state = CorridorState.TRAVERSE
        if self.state == CorridorState.TRAVERSE and progress >= max(8.0, length - 12.0):
            self.state = CorridorState.EXIT
        if self.state == CorridorState.EXIT and progress >= length + 12.0:
            self.state = CorridorState.RECOVER
        if self.state == CorridorState.RECOVER:
            self.state = CorridorState.COMPLETE
        return CorridorCommand(self.state, exit_point.copy(), target_velocity,
                               desired_heading, rotation, False)

