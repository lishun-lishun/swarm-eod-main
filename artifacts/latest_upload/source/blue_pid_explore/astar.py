"""Risk-aware eight-neighbour A* on an occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np

from .occupancy_map import Cell


@dataclass(frozen=True)
class AStarResult:
    path: tuple[tuple[int, int], ...]
    cost: float
    expanded: int
    success: bool


def astar(grid: np.ndarray, start: Sequence[int], goal: Sequence[int], *,
          obstacle_cost: np.ndarray | None = None, unknown_cost: float = 3.0,
          allow_unknown: bool = False) -> AStarResult:
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    h, w = grid.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= goal[0] < w and 0 <= goal[1] < h):
        return AStarResult((), math.inf, 0, False)
    blocked = {int(Cell.OCCUPIED), int(Cell.INFLATED)}
    if int(grid[goal[1], goal[0]]) in blocked:
        return AStarResult((), math.inf, 0, False)
    queue = [(0.0, 0.0, start)]
    costs = {start: 0.0}; parents = {start: None}; expanded = 0
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
             (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > costs.get(node, math.inf) + 1e-9:
            continue
        expanded += 1
        if node == goal:
            path = []
            while node is not None:
                path.append(node); node = parents[node]
            return AStarResult(tuple(reversed(path)), cost, expanded, True)
        for dx, dy, length in moves:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
            value = int(grid[nxt[1], nxt[0]])
            if value in blocked or (value == int(Cell.UNKNOWN) and not allow_unknown):
                continue
            risk = 0.0 if obstacle_cost is None else float(obstacle_cost[nxt[1], nxt[0]])
            extra_unknown = unknown_cost if value == int(Cell.UNKNOWN) else 0.0
            tentative = cost + length + risk + extra_unknown
            if tentative >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = tentative; parents[nxt] = node
            heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
            heapq.heappush(queue, (tentative + heuristic, tentative, nxt))
    return AStarResult((), math.inf, expanded, False)


def simplify_path(path: Sequence[Sequence[int]], traversable) -> tuple[tuple[int, int], ...]:
    if len(path) <= 2:
        return tuple(tuple(map(int, p)) for p in path)
    result = [tuple(path[0])]; index = 0
    while index < len(path) - 1:
        furthest = index + 1
        for candidate in range(index + 2, len(path)):
            if traversable(tuple(path[index]), tuple(path[candidate])):
                furthest = candidate
            else:
                break
        result.append(tuple(path[furthest])); index = furthest
    return tuple(result)


