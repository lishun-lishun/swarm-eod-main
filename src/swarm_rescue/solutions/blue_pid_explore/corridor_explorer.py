"""Committed long-corridor exploration using only legal local sensors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .nonlinear_braking import NonlinearStoppingModel
from .oriented_footprint import OrientedFootprint, Pose2, wrap_angle


class CorridorExploreState(str, Enum):
    IDLE = "IDLE"
    ENTER = "ENTER"
    ADVANCE = "ADVANCE"
    END_APPROACH = "END_APPROACH"
    INSPECT = "INSPECT"
    BRANCH = "BRANCH"
    CLOSED = "CLOSED"
    COMPLETE = "COMPLETE"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class CorridorExploreOutput:
    active: bool
    state: CorridorExploreState
    target_velocity: np.ndarray
    target_heading: float
    end_gap: float
    safe_gap: float
    corridor_width: float
    branch_angle: float | None
    reason: str


class CorridorExplorer:
    """Commit to a sensed corridor until its end and branches are inspected.

    LiDAR distances are treated as center-origin ranges.  Surface clearance is
    therefore range minus the convex-hull support in that ray direction,
    exactly once.  No map coordinates or map-specific constants are used.
    """

    def __init__(
        self,
        footprint: OrientedFootprint,
        *,
        max_range: float = 230.0,
        target_gap: float = 2.0,
        geometry_margin: float = 3.0,
        sigma_gain: float = 2.0,
        lidar_sigma: float = 2.5,
        corridor_min_length: float = 130.0,
        corridor_max_width: float = 130.0,
        branch_margin: float = 4.0,
        confirm_frames: int = 3,
        entry_confirm_frames: int = 8,
        inspect_frames: int = 10,
        task_ttl: int = 850,
        advance_speed: float = 2.8,
    ):
        self.footprint = footprint
        self.max_range = float(max_range)
        self.target_gap = float(target_gap)
        self.geometry_margin = float(geometry_margin)
        self.sigma_gain = float(sigma_gain)
        self.lidar_sigma = float(lidar_sigma)
        self.corridor_min_length = float(corridor_min_length)
        self.corridor_max_width = float(corridor_max_width)
        self.branch_margin = float(branch_margin)
        self.confirm_frames = int(confirm_frames)
        self.entry_confirm_frames = int(entry_confirm_frames)
        self.inspect_frames = int(inspect_frames)
        self.task_ttl = int(task_ttl)
        self.advance_speed = float(advance_speed)
        self.stopping = NonlinearStoppingModel(physics_substeps=10)
        self.state = CorridorExploreState.IDLE
        self.direction = 0.0
        self.started_step = -1
        self.start_position = np.zeros(2)
        self.end_position: np.ndarray | None = None
        self._detect_count = 0
        self._detect_direction: float | None = None
        self._settled_count = 0
        self._inspect_count = 0
        self._lost_walls = 0
        self._branch_counts: dict[int, int] = {}
        self._branch_aligned_count = 0
        self.branch_angle: float | None = None
        self._branch_start: np.ndarray | None = None
        self.completed_endpoints: list[tuple[np.ndarray, int]] = []
        self.stats = {
            "entered": 0,
            "ends_reached": 0,
            "closed_ends": 0,
            "branches_found": 0,
            "branches_entered": 0,
            "completed": 0,
            "paused": 0,
            "midway_reversals_blocked": 0,
            "min_end_gap": math.inf,
            "min_observation_gap": math.inf,
            "approach_failures": 0,
        }

    @property
    def committed(self) -> bool:
        return self.state not in (
            CorridorExploreState.IDLE,
            CorridorExploreState.COMPLETE,
            CorridorExploreState.PAUSED,
        )

    def reset(self) -> None:
        self.state = CorridorExploreState.IDLE
        self._detect_count = self._settled_count = self._inspect_count = 0
        self._detect_direction = None
        self._lost_walls = 0
        self._branch_counts.clear()
        self._branch_aligned_count = 0
        self.branch_angle = None
        self._branch_start = None

    def pause(self) -> None:
        if self.committed:
            self.state = CorridorExploreState.PAUSED
            self.stats["paused"] += 1

    def resume(self, position: Sequence[float], step: int) -> bool:
        if self.state != CorridorExploreState.PAUSED:
            return False
        if step - self.started_step > self.task_ttl:
            self.reset()
            return False
        reference = self.end_position if self.end_position is not None else self.start_position
        if float(np.linalg.norm(np.asarray(position, float) - reference)) > 180.0:
            return False
        self.state = CorridorExploreState.ADVANCE
        return True

    def _rays(self, distances, angles, heading: float):
        d = np.asarray(distances, dtype=float).reshape(-1)
        a = np.asarray(angles, dtype=float).reshape(-1)
        n = min(len(d), len(a))
        valid = np.isfinite(d[:n]) & (d[:n] > 0.0)
        return d[:n], a[:n], valid

    @staticmethod
    def _cone_values(d, relative, valid, center: float, half: float):
        error = np.abs((relative - center + math.pi) % (2.0 * math.pi) - math.pi)
        return d[valid & (error <= half)]

    def _geometry(self, distances, angles, heading: float, travel_heading: float):
        d, a, valid = self._rays(distances, angles, heading)
        relative = (a - (travel_heading - heading) + math.pi) % (2.0 * math.pi) - math.pi
        left = self._cone_values(d, relative, valid, math.pi / 2.0, 0.20)
        right = self._cone_values(d, relative, valid, -math.pi / 2.0, 0.20)
        front = self._cone_values(d, relative, valid, 0.0, 0.13)
        left_wall = left[left < self.max_range - 9.0]
        right_wall = right[right < self.max_range - 9.0]
        left_range = float(np.percentile(left_wall, 35)) if left_wall.size >= 3 else math.inf
        right_range = float(np.percentile(right_wall, 35)) if right_wall.size >= 3 else math.inf
        front_range = float(np.percentile(front, 25)) if front.size else self.max_range
        corridor_width = left_range + right_range
        lateral = np.array([-math.sin(travel_heading), math.cos(travel_heading)])
        footprint_width = self.footprint.projection_width(heading, lateral)
        side_consistent = (
            math.isfinite(left_range) and math.isfinite(right_range)
            and corridor_width >= footprint_width + 2.0 * self.branch_margin
            and corridor_width <= self.corridor_max_width
        )
        return side_consistent, corridor_width, front_range, footprint_width, d, relative, valid

    def _support(self, heading: float, direction: float) -> float:
        normal = np.array([math.cos(direction), math.sin(direction)])
        points = self.footprint.transformed(Pose2(0.0, 0.0, heading))
        return float(np.max(points @ normal))

    def _safe_gap(self, normal_speed: float) -> float:
        robust_speed = max(0.0, float(normal_speed)) + 0.25
        return max(
            self.target_gap,
            self.geometry_margin + self.sigma_gain * self.lidar_sigma
            + self.stopping.stopping_distance(robust_speed),
        )

    def _opening_candidates(self, d, relative, valid, heading: float):
        """Return continuous side openings wide enough for the real hull."""
        order = np.argsort(relative)
        rel, ranges, mask = relative[order], d[order], valid[order]
        open_mask = mask & (ranges >= self.max_range - 10.0)
        candidates = []
        start = None
        for index in range(len(rel) + 1):
            is_open = index < len(rel) and bool(open_mask[index])
            if is_open and start is None:
                start = index
            if (not is_open or index == len(rel)) and start is not None:
                end = index
                count = end - start
                center = float(np.mean(rel[start:end]))
                span = float(rel[end - 1] - rel[start]) if count > 1 else 0.0
                # Only lateral/forward-lateral gaps are new branches; the
                # rear opening is the already inspected corridor entrance.
                if count >= 5 and 0.38 <= abs(center) <= 2.25:
                    probe_range = min(self.max_range - 10.0, 120.0)
                    width = 2.0 * probe_range * math.sin(max(0.0, span) / 2.0)
                    branch_world = wrap_angle(self.direction + center)
                    normal = np.array([-math.sin(branch_world), math.cos(branch_world)])
                    required = self.footprint.projection_width(heading, normal,
                                                                extra_margin=self.branch_margin)
                    if width >= required:
                        candidates.append((branch_world, width, count))
                start = None
        return candidates

    def _output(self, velocity, heading, *, active=True, end_gap=math.inf,
                safe_gap=math.inf, width=math.inf, reason=""):
        return CorridorExploreOutput(
            active, self.state, np.asarray(velocity, dtype=float), float(heading),
            float(end_gap), float(safe_gap), float(width), self.branch_angle, reason,
        )

    def update(
        self,
        *,
        step: int,
        position: Sequence[float],
        velocity: Sequence[float],
        heading: float,
        travel_heading: float,
        lidar_distances,
        lidar_angles,
    ) -> CorridorExploreOutput:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if lidar_distances is None or lidar_angles is None:
            return self._output((0.0, 0.0), heading, active=False, reason="NO_LIDAR")
        if self.state == CorridorExploreState.PAUSED:
            return self._output((0.0, 0.0), heading, active=False, reason="PAUSED")
        if self.committed and step - self.started_step > self.task_ttl:
            self.stats["approach_failures"] += 1
            self.reset()
        direction = self.direction if self.committed else float(travel_heading)
        side_ok, width, front_range, _, d, relative, valid = self._geometry(
            lidar_distances, lidar_angles, heading, direction,
        )
        if not self.committed:
            recently_done = any(
                step < expiry and np.linalg.norm(position - endpoint) < 70.0
                for endpoint, expiry in self.completed_endpoints
            )
            travel_forward = np.array([math.cos(travel_heading), math.sin(travel_heading)])
            along_speed = float(velocity @ travel_forward)
            speed = float(np.linalg.norm(velocity))
            aligned = along_speed >= 0.8 and along_speed >= 0.88 * max(speed, 1e-6)
            direction_stable = (
                self._detect_direction is None
                or abs(wrap_angle(travel_heading - self._detect_direction)) <= 0.14
            )
            detect = (
                side_ok and front_range >= self.corridor_min_length
                and aligned and direction_stable and not recently_done
            )
            self._detect_count = self._detect_count + 1 if detect else 0
            self._detect_direction = float(travel_heading) if detect else None
            if self._detect_count < self.entry_confirm_frames:
                self.state = CorridorExploreState.IDLE
                return self._output((0.0, 0.0), travel_heading, active=False,
                                    width=width, reason="NO_COMMIT")
            self.state = CorridorExploreState.ENTER
            self.direction = float(travel_heading)
            self.started_step = int(step)
            self.start_position = position.copy()
            self.stats["entered"] += 1
            direction = self.direction

        forward = np.array([math.cos(direction), math.sin(direction)])
        normal_speed = max(0.0, float(velocity @ forward))
        support = self._support(heading, direction)
        end_gap = max(-support, front_range - support)
        safe_gap = self._safe_gap(normal_speed)
        self.stats["min_end_gap"] = min(self.stats["min_end_gap"], end_gap)

        if self.state == CorridorExploreState.ENTER:
            self.state = CorridorExploreState.ADVANCE

        if self.state == CorridorExploreState.ADVANCE:
            self._lost_walls = 0 if side_ok else self._lost_walls + 1
            if end_gap <= safe_gap + 34.0:
                self.state = CorridorExploreState.END_APPROACH
            elif self._lost_walls > 7:
                progressed = float((position - self.start_position) @ forward)
                if progressed < 45.0:
                    # A few nearby objects can mimic two corridor walls for a
                    # moment.  Do not turn that short-lived geometry into an
                    # intersection task.
                    self.stats["approach_failures"] += 1
                    self.reset()
                    return self._output(np.zeros(2), heading, active=False,
                                        end_gap=end_gap, safe_gap=safe_gap,
                                        width=width, reason="FALSE_CORRIDOR")
                # A widening is itself an intersection. Stop and inspect it;
                # never silently release the commitment to frontier scoring.
                self.state = CorridorExploreState.INSPECT
                self._inspect_count = 0
            else:
                return self._output(self.advance_speed * forward, direction, end_gap=end_gap,
                                    safe_gap=safe_gap, width=width, reason="ADVANCE")

        if self.state == CorridorExploreState.END_APPROACH:
            speed = float(np.clip(0.22 * (end_gap - safe_gap), 0.0, 1.35))
            if end_gap <= safe_gap + 1.8 and normal_speed <= 0.45:
                self._settled_count += 1
            else:
                self._settled_count = 0
            if self._settled_count >= 3:
                self.state = CorridorExploreState.INSPECT
                self._inspect_count = 0
                self.end_position = position.copy()
                self.stats["ends_reached"] += 1
                self.stats["min_observation_gap"] = min(
                    self.stats["min_observation_gap"], end_gap
                )
            else:
                return self._output(speed * forward, direction, end_gap=end_gap,
                                    safe_gap=safe_gap, width=width, reason="END_APPROACH")

        if self.state == CorridorExploreState.INSPECT:
            self._inspect_count += 1
            seen_keys = set()
            for angle, gap_width, _ in self._opening_candidates(d, relative, valid, heading):
                key = int(round(angle / math.radians(10.0)))
                seen_keys.add(key)
                self._branch_counts[key] = self._branch_counts.get(key, 0) + 1
                if self._branch_counts[key] >= self.confirm_frames:
                    self.branch_angle = angle
                    self.state = CorridorExploreState.BRANCH
                    self._branch_start = position.copy()
                    self._branch_aligned_count = 0
                    self.stats["branches_found"] += 1
                    break
            for key in list(self._branch_counts):
                if key not in seen_keys:
                    # Confirmation must be consecutive; isolated/noisy rays
                    # from older frames are not accumulated indefinitely.
                    self._branch_counts[key] = 0
            if self.state == CorridorExploreState.INSPECT:
                if self._inspect_count < self.inspect_frames:
                    # LiDAR is already 360 degrees: hold position, no pointless
                    # scan rotation. Small drift is actively removed upstream.
                    return self._output(np.zeros(2), heading, end_gap=end_gap,
                                        safe_gap=safe_gap, width=width, reason="INSPECT")
                self.state = CorridorExploreState.CLOSED
                self.stats["closed_ends"] += 1

        if self.state == CorridorExploreState.BRANCH:
            assert self.branch_angle is not None
            branch_forward = np.array([math.cos(self.branch_angle), math.sin(self.branch_angle)])
            finite_ranges = d[valid]
            rotation_clearance = (
                float(np.percentile(finite_ranges, 5)) if finite_ranges.size else 0.0
            )
            required_rotation_clearance = self.footprint.radius + 4.0
            if rotation_clearance < required_rotation_clearance:
                # A pure yaw sweep occupies the footprint circumcircle.  Back
                # away from the end wall before rotating into a side branch;
                # rotating in place at the 2 px observation gap was a major
                # collision source in Map09.
                return self._output(-0.75 * forward, direction,
                                    end_gap=end_gap, safe_gap=safe_gap, width=width,
                                    reason="BRANCH_CLEARANCE")
            heading_error = abs(wrap_angle(self.branch_angle - heading))
            if heading_error > 0.18:
                self._branch_aligned_count = 0
                return self._output(np.zeros(2), self.branch_angle,
                                    end_gap=end_gap, safe_gap=safe_gap, width=width,
                                    reason="BRANCH_ALIGN")
            self._branch_aligned_count += 1
            if self._branch_aligned_count < 2:
                return self._output(np.zeros(2), self.branch_angle,
                                    end_gap=end_gap, safe_gap=safe_gap, width=width,
                                    reason="BRANCH_ALIGN")
            progressed = 0.0 if self._branch_start is None else float(
                (position - self._branch_start) @ branch_forward
            )
            if progressed >= 55.0:
                self.stats["branches_entered"] += 1
                self.state = CorridorExploreState.COMPLETE
            else:
                return self._output(1.8 * branch_forward, self.branch_angle,
                                    end_gap=end_gap, safe_gap=safe_gap, width=width,
                                    reason="BRANCH_ENTER")

        if self.state in (CorridorExploreState.CLOSED, CorridorExploreState.COMPLETE):
            endpoint = position.copy()
            self.completed_endpoints.append((endpoint, step + 1200))
            self.completed_endpoints = self.completed_endpoints[-24:]
            self.stats["completed"] += 1
            result = self._output(np.zeros(2), heading, active=False, end_gap=end_gap,
                                  safe_gap=safe_gap, width=width, reason=self.state.value)
            self.state = CorridorExploreState.IDLE
            self._detect_count = 0
            return result
        return self._output(np.zeros(2), heading, active=False, reason="IDLE")

