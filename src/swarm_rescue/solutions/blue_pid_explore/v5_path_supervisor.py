"""V5: detect nonexecuting local waypoints and false center-field arrival.

This module only recommends recovery; it cannot bypass payload geometry,
LiDAR safety, legal sensor limits, or official disposal confirmation.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import dist, isfinite
from typing import Sequence


@dataclass(frozen=True)
class PathSignal:
    action: str  # CONTINUE, REPLAN, TERMINAL_CONFLICT, RECOVER
    reason: str
    displacement_px: float
    progress_px: float


class PathProgressSupervisor:
    def __init__(self, *, window_frames: int = 75,
                 min_progress_px: float = 12.0,
                 min_motion_px: float = 8.0,
                 terminal_mismatch_px: float = 85.0,
                 cooldown_frames: int = 40):
        self.window = int(window_frames)
        self.min_progress = float(min_progress_px)
        self.min_motion = float(min_motion_px)
        self.mismatch = float(terminal_mismatch_px)
        self.cooldown = int(cooldown_frames)
        self._samples = deque(maxlen=max(2, self.window + 1))
        self._tag = None
        self._last_emit = -10**9
        self._conflict_frames = 0
        self._replan_attempts = 0

    def reset(self) -> None:
        self._samples.clear()
        self._tag = None
        self._conflict_frames = 0
        self._replan_attempts = 0

    def observe(self, *, step: int, tag: str, position: Sequence[float],
                path_cost_px: float | None, waypoint_distance_px: float | None,
                path_points: int, is_carrying: bool,
                live_center_distance_px: float | None) -> PathSignal:
        p = (float(position[0]), float(position[1]))
        if not all(map(isfinite, p)):
            return PathSignal('REPLAN', 'invalid_pose', 0.0, 0.0)
        if self._tag != tag:
            self.reset()
            self._tag = tag
        d = (float(path_cost_px) if path_cost_px is not None
             and isfinite(path_cost_px) else None)
        self._samples.append((int(step), p, d))
        disp = 0.0
        advance = 0.0
        if len(self._samples) >= 2:
            first = self._samples[0]
            disp = dist(first[1], p)
            if d is not None and first[2] is not None:
                advance = first[2] - d
        far_center = (is_carrying and live_center_distance_px is not None
                      and isfinite(live_center_distance_px)
                      and live_center_distance_px >= self.mismatch)
        false_terminal = (far_center and d is not None and d <= 5.0
                          and path_points <= 1)
        self._conflict_frames = (self._conflict_frames + 1 if false_terminal else 0)
        if step - self._last_emit < self.cooldown:
            return PathSignal('CONTINUE', 'cooldown', disp, advance)
        if self._conflict_frames >= 6:
            self._last_emit = step
            return PathSignal('TERMINAL_CONFLICT', 'map_zero_semantic_far', disp, advance)
        # Path cost is still substantial but no route was produced.
        if path_points == 0 and d is not None and d > 25.0:
            self._last_emit = step
            self._replan_attempts += 1
            action = 'RECOVER' if self._replan_attempts >= 3 else 'REPLAN'
            return PathSignal(action, 'nonzero_cost_empty_path', disp, advance)
        if len(self._samples) >= 2 and step - self._samples[0][0] >= self.window:
            if disp < self.min_motion or advance < self.min_progress:
                self._last_emit = step
                self._replan_attempts += 1
                action = 'RECOVER' if self._replan_attempts >= 3 else 'REPLAN'
                return PathSignal(action, 'insufficient_route_progress', disp, advance)
            self._replan_attempts = 0
            # Reset the progress baseline after each completed window.
            self._samples.clear()
            self._samples.append((int(step), p, d))
        return PathSignal('CONTINUE', 'progress_ok', disp, advance)


def reachable_lookahead(position: Sequence[float], path: Sequence[Sequence[float]],
                        segment_safe, *, max_lookahead_px: float = 75.0):
    """Return farthest sequential waypoint that is verified locally traversable.

    Caller MUST supply a segment_safe(p, q) check for current payload hull,
    local obstacles, unknown cells and pose uncertainty. Never skip a wall.
    """
    if not path:
        return None
    p = (float(position[0]), float(position[1]))
    best = None
    for waypoint in path:
        q = (float(waypoint[0]), float(waypoint[1]))
        if dist(p, q) > max_lookahead_px:
            break
        if not segment_safe(p, q):
            break
        best = q
    return best
