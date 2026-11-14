"""Local lattice A* in (x, y, heading) with swept-hull validation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np

from .oriented_footprint import OrientedFootprint, Pose2, angle_delta, wrap_angle


@dataclass(frozen=True)
class SE2PlanResult:
    poses: tuple[Pose2, ...]
    cost: float
    expanded: int
    success: bool
    reason: str


class SE2LocalPlanner:
    def __init__(
        self, footprint: OrientedFootprint, *, heading_bins: int = 32,
        rotation_cost: float = 8.0, risk_weight: float = 2.0,
        reverse_cost: float = 0.25, max_expansions: int = 30000,
    ):
        if heading_bins < 8:
            raise ValueError("heading_bins must be >= 8")
        self.footprint = footprint
        self.heading_bins = int(heading_bins)
        self.angle_step = 2.0 * math.pi / self.heading_bins
        self.rotation_cost = float(rotation_cost)
        self.risk_weight = float(risk_weight)
        self.reverse_cost = float(reverse_cost)
        self.max_expansions = int(max_expansions)

    def _heading(self, index: int) -> float:
        return wrap_angle(-math.pi + (index % self.heading_bins) * self.angle_step)

    def _heading_index(self, angle: float) -> int:
        return int(round((wrap_angle(angle) + math.pi) / self.angle_step)) % self.heading_bins

    @staticmethod
    def _heuristic(x: int, y: int, h: int, goal: tuple[int, int, int], resolution: float, angle_step: float):
        distance = resolution * math.hypot(goal[0] - x, goal[1] - y)
        heading = min((h - goal[2]) % round(2 * math.pi / angle_step),
                      (goal[2] - h) % round(2 * math.pi / angle_step)) * angle_step
        return distance + 2.0 * heading

    def plan(
        self, start: Pose2, goal: Pose2, occupied: np.ndarray,
        origin: Sequence[float], resolution: float,
        *, risk: np.ndarray | None = None,
    ) -> SE2PlanResult:
        origin = np.asarray(origin, dtype=float)
        h_grid, w_grid = occupied.shape

        def state(pose: Pose2):
            grid = np.floor((pose.position - origin) / resolution).astype(int)
            return int(grid[0]), int(grid[1]), self._heading_index(pose.theta)

        start_state, goal_state = state(start), state(goal)
        if self.footprint.collides_grid(start, occupied, origin, resolution):
            return SE2PlanResult((), math.inf, 0, False, "START_COLLISION")
        if self.footprint.collides_grid(goal, occupied, origin, resolution):
            return SE2PlanResult((), math.inf, 0, False, "GOAL_COLLISION")
        queue = [(0.0, 0.0, start_state)]
        costs = {start_state: 0.0}
        parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_state: None}
        expanded = 0
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while queue and expanded < self.max_expansions:
            _, current_cost, current = heapq.heappop(queue)
            if current_cost > costs.get(current, math.inf) + 1e-9:
                continue
            expanded += 1
            if current[:2] == goal_state[:2] and current[2] == goal_state[2]:
                path = []
                node = current
                while node is not None:
                    pos = origin + resolution * (np.array(node[:2], dtype=float) + 0.5)
                    path.append(Pose2(float(pos[0]), float(pos[1]), self._heading(node[2])))
                    node = parents[node]
                path.reverse()
                path[0] = start; path[-1] = goal
                return SE2PlanResult(tuple(path), current_cost, expanded, True, "OK")
            x, y, heading_index = current
            current_pose = Pose2(*(origin + resolution * (np.array([x, y]) + 0.5)), self._heading(heading_index))
            neighbors: list[tuple[tuple[int, int, int], float]] = [
                ((x, y, (heading_index - 1) % self.heading_bins), self.rotation_cost * self.angle_step),
                ((x, y, (heading_index + 1) % self.heading_bins), self.rotation_cost * self.angle_step),
            ]
            for dx, dy in moves:
                distance = math.hypot(dx, dy)
                travel_angle = math.atan2(dy, dx)
                alignment = abs(angle_delta(travel_angle, current_pose.theta))
                reverse_penalty = self.reverse_cost if alignment > math.pi / 2 else 0.0
                neighbors.append(((x + dx, y + dy, heading_index), resolution * distance + reverse_penalty))
            for neighbor, edge_cost in neighbors:
                nx, ny, nh = neighbor
                if nx < 0 or ny < 0 or nx >= w_grid or ny >= h_grid:
                    continue
                next_pose = Pose2(*(origin + resolution * (np.array([nx, ny]) + 0.5)), self._heading(nh))
                if self.footprint.swept_collides(current_pose, next_pose, occupied, origin, resolution):
                    continue
                risk_cost = 0.0 if risk is None else self.risk_weight * float(risk[ny, nx])
                tentative = current_cost + edge_cost + risk_cost
                if tentative + 1e-9 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = tentative
                parents[neighbor] = current
                heuristic = self._heuristic(nx, ny, nh, goal_state, resolution, self.angle_step)
                heapq.heappush(queue, (tentative + heuristic, tentative, neighbor))
        reason = "MAX_EXPANSIONS" if expanded >= self.max_expansions else "NO_PATH"
        return SE2PlanResult((), math.inf, expanded, False, reason)

