"""Occupancy/frontier/A*/SE(2)/lookahead navigation without actuator access."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .astar import astar
from .frontier_explorer import Frontier, FrontierExplorer
from .occupancy_map import Cell, OccupancyMap
from .oriented_footprint import OrientedFootprint, Pose2
from .path_tracker import LookaheadPathTracker, PathTrackOutput
from .se2_local_planner import SE2LocalPlanner
from .semantic_coverage import SemanticCoverageMap


@dataclass(frozen=True)
class NavigationOutput:
    target_velocity: np.ndarray
    target_heading: float
    path: tuple[np.ndarray, ...]
    reached: bool
    replan_required: bool
    used_se2: bool
    reason: str


class NavigationManager:
    def __init__(self, size_world: Sequence[float], footprint: OrientedFootprint,
                 *, resolution: float = 8.0):
        self.map = OccupancyMap(size_world, resolution=resolution)
        self.semantic = SemanticCoverageMap(self.map)
        self.frontiers = FrontierExplorer()
        self.tracker = LookaheadPathTracker()
        self.footprint = footprint
        self.se2 = SE2LocalPlanner(footprint, heading_bins=16, max_expansions=12000)
        self.path: tuple[np.ndarray, ...] = ()
        self.pose_path: tuple[Pose2, ...] = ()
        self.goal: np.ndarray | None = None
        self.last_plan_step = -1000
        self.used_se2 = False

    def update_sensors(self, position, heading, lidar_distances, lidar_angles,
                       semantic_angles) -> None:
        self.map.update_scan(position, heading, lidar_distances, lidar_angles)
        blocking = self.map.cells() == int(Cell.OCCUPIED)
        self.semantic.update(position, heading, semantic_angles, blocking_cells=blocking)

    def _nearest_free(self, cell, grid):
        x, y = cell
        if self.map.inside(x, y) and grid[y, x] == int(Cell.FREE):
            return x, y
        queue = deque([(x, y)]); seen = {(x, y)}
        while queue and len(seen) < 500:
            cx, cy = queue.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen or not self.map.inside(nx, ny): continue
                if grid[ny, nx] == int(Cell.FREE): return nx, ny
                seen.add((nx, ny)); queue.append((nx, ny))
        return None

    def _distance_field(self, start, grid):
        dist = np.full(grid.shape, np.inf, dtype=float)
        if start is None: return dist
        queue=deque([start]); dist[start[1],start[0]]=0.0
        while queue:
            x,y=queue.popleft()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=x+dx,y+dy
                if self.map.inside(nx,ny) and grid[ny,nx]==int(Cell.FREE) and not np.isfinite(dist[ny,nx]):
                    dist[ny,nx]=dist[y,x]+1; queue.append((nx,ny))
        return dist

    def choose_frontier(self, position, peer_goals: Sequence[Sequence[float]] = ()) -> Frontier | None:
        grid = self.map.cells(inflation_radius=self.footprint.radius + 2.0)
        start = self._nearest_free(self.map.world_to_grid(position), grid)
        distances = self._distance_field(start, grid)
        candidates = self.frontiers.frontiers(grid, start or (0,0), distances)
        if not candidates: return None
        peer_cells = [self.map.world_to_grid(p) for p in peer_goals]
        def adjusted(frontier):
            separation = min((math.hypot(frontier.representative[0]-p[0], frontier.representative[1]-p[1])
                              for p in peer_cells), default=100.0)
            return frontier.score + min(15.0, 0.5 * separation)
        return max(candidates, key=adjusted)

    def _plan(self, position, heading, target, step):
        grid = self.map.cells(inflation_radius=self.footprint.radius + 2.0)
        start = self._nearest_free(self.map.world_to_grid(position), grid)
        goal = self._nearest_free(self.map.world_to_grid(target), grid)
        if start is None or goal is None:
            self.path=(); self.pose_path=(); return "NO_FREE_ENDPOINT"
        result=astar(grid,start,goal)
        if result.success:
            self.path=tuple(self.map.grid_to_world(cell) for cell in result.path)
            self.pose_path=(); self.used_se2=False; self.tracker.reset(); self.last_plan_step=step
            return "A_STAR"
        # SE(2) is a local narrow-passage planner, never a whole-map fallback.
        # A full-map 16-layer search would violate the real-time budget.
        if np.linalg.norm(np.asarray(target)-np.asarray(position))>260.0:
            self.path=(); self.pose_path=(); self.last_plan_step=step
            return "BLOCKED:SE2_OUT_OF_LOCAL_RANGE"
        # Plan orientation-aware on observed occupancy (unknown remains blocked).
        raw=self.map.cells(inflation_radius=0.0)
        occupied=raw!=int(Cell.FREE)
        se2=self.se2.plan(Pose2(float(position[0]),float(position[1]),float(heading)),
                          Pose2(float(target[0]),float(target[1]),float(heading)),
                          occupied,self.map.origin,self.map.resolution)
        if se2.success:
            self.pose_path=se2.poses; self.path=tuple(p.position for p in se2.poses)
            self.used_se2=True; self.tracker.reset(); self.last_plan_step=step
            return "SE2"
        self.path=(); self.pose_path=(); self.last_plan_step=step
        return f"BLOCKED:{se2.reason}"

    def navigate(self, position, velocity, heading, target, step, *, force_replan=False) -> NavigationOutput:
        target=np.asarray(target,float)
        changed=self.goal is None or np.linalg.norm(target-self.goal)>8.0
        if changed: self.goal=target.copy(); force_replan=True
        if force_replan or not self.path or step-self.last_plan_step>=45:
            reason=self._plan(position,heading,target,step)
        else: reason="TRACK"
        if not self.path:
            return NavigationOutput(np.zeros(2),heading,(),False,True,self.used_se2,reason)
        tracked=self.tracker.compute(position,velocity,self.path)
        target_heading=heading
        if self.used_se2 and self.pose_path:
            index=min(tracked.path_index,len(self.pose_path)-1); target_heading=self.pose_path[index].theta
        elif np.linalg.norm(tracked.target_velocity)>0.2:
            target_heading=math.atan2(tracked.target_velocity[1],tracked.target_velocity[0])
        return NavigationOutput(tracked.target_velocity,target_heading,self.path,tracked.reached,
                                False,self.used_se2,reason)

