"""Global-ranking, local-execution frontiers for existing GridMap/MCP.

Only uses measured/fused occupancy, clearance, current Dijkstra distance,
and optional conservative semantic-ray evidence. Never reads true map/bombs.
The returned CELL is a *known-free goal*, not a traversal permission.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import heapq
from collections import deque
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MacroGoal:
    cell: tuple[int, int]
    xy: tuple[float, float]
    region: tuple[int, int]
    score: float
    route_cost_px: float
    unknown_nearby: float
    far_unknown: float
    route: tuple[tuple[float, float], ...]
    expansions: int


class MacroFrontierPlanner:
    def __init__(self, *, cell_px=10.0, region_cells=20,
                 min_clearance_px=18.0, max_route_px=4000.0,
                 max_expansions=46000):
        self.cell_px = float(cell_px)
        self.region_cells = int(region_cells)
        self.min_clearance = float(min_clearance_px)
        self.max_route_px = float(max_route_px)
        self.max_expansions = int(max_expansions)
        self.history = deque(maxlen=32)  # (region, step, observed_gain)
        self.cooldowns: dict[tuple[int, int], int] = {}
        if self.cell_px <= 0 or self.region_cells < 4:
            raise ValueError('invalid spatial resolution')

    def mark_result(self, region, now, *, new_semantic_cells=0,
                    reached=False, blocked=False):
        region = tuple(map(int, region))
        self.history.append((region, int(now), int(new_semantic_cells)))
        if blocked or (reached and new_semantic_cells <= 0):
            self.cooldowns[region] = int(now) + (450 if blocked else 210)

    def select(self, *, occ, clearance_px, current_xy, map_origin_xy,
               local_start_safe, now: int, semantic_seen_at=None,
               own_evidence=None, peer_goal_xy=(),
               exclude_near_xy=(), max_candidates=180):
        occ = np.asarray(occ, dtype=float)
        clr = np.asarray(clearance_px, dtype=float)
        if occ.ndim != 2 or occ.shape != clr.shape:
            raise ValueError('inconsistent GridMap fields')
        if semantic_seen_at is not None and np.asarray(semantic_seen_at).shape != occ.shape:
            raise ValueError('semantic evidence shape mismatch')
        if own_evidence is not None and np.asarray(own_evidence).shape != occ.shape:
            raise ValueError('own evidence shape mismatch')
        if local_start_safe is None:
            raise ValueError('a real local LiDAR start connector is required')
        free = (occ < -0.6) & (clr >= self.min_clearance) & np.isfinite(clr)
        unknown = (occ >= -0.6) & (occ <= 1.0)
        if own_evidence is not None:
            # Conflicting direct/free and fused/blocked evidence demands a closer view;
            # it is NEVER reclassified as traversable.
            unknown |= (np.asarray(own_evidence) < -0.8) & (occ >= -0.6)
        adjacent = np.zeros_like(unknown)
        adjacent[1:] |= unknown[:-1]
        adjacent[:-1] |= unknown[1:]
        adjacent[:, 1:] |= unknown[:, :-1]
        adjacent[:, :-1] |= unknown[:, 1:]
        frontier = free & adjacent
        if not np.any(frontier):
            return None
        ny, nx = free.shape
        def world(cell):
            y,x=cell
            return (float(map_origin_xy[0])+(x+.5)*self.cell_px,
                    float(map_origin_xy[1])+(y+.5)*self.cell_px)
        sx = int(math.floor((float(current_xy[0])-float(map_origin_xy[0]))/self.cell_px))
        sy = int(math.floor((float(current_xy[1])-float(map_origin_xy[1]))/self.cell_px))
        starts=[]
        for radius in range(4):
            for y in range(max(0,sy-radius),min(ny,sy+radius+1)):
                for x in range(max(0,sx-radius),min(nx,sx+radius+1)):
                    if free[y,x] and local_start_safe(tuple(current_xy),world((y,x))):
                        starts.append((math.dist(current_xy,world((y,x))),(y,x)))
            if starts:
                break
        if not starts:
            return None
        connector,start=min(starts)
        # One bounded, FOUR-connected Dijkstra, never through UNKNOWN or
        # diagonally through a wall corner. No invalid legacy cost-field reuse.
        dist=np.full((ny,nx),np.inf,dtype=float)
        dist[start]=connector
        parents={}
        queue=[(connector,start)]
        expanded=0
        while queue and expanded<self.max_expansions:
            cost,c=heapq.heappop(queue)
            if cost>dist[c]+1e-9:
                continue
            expanded+=1
            if cost>self.max_route_px:
                break
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(c[0]+dy,c[1]+dx)
                if not (0<=n[0]<ny and 0<=n[1]<nx and free[n]):
                    continue
                # Smooth narrow passages are costly, never silently open.
                new=cost+self.cell_px*(1.+0.25*max(0.,30.-clr[n])/30.)
                if new+1e-9<dist[n]:
                    dist[n]=new
                    parents[n]=c
                    heapq.heappush(queue,(new,n))
        iy,ix=np.nonzero(frontier & np.isfinite(dist) & (dist<=self.max_route_px))
        if not len(iy):
            return None
        route=dist[iy,ix]
        # TWO spatial horizons: local wall lips and global unmapped wings.
        # Remote frontiers are not discarded in favor of first 12 near frontiers.
        u = unknown.astype(np.float32)
        near_mass = ndimage.uniform_filter(u, size=17, mode='constant') * (17 * 17)
        far_mass = ndimage.uniform_filter(u, size=41, mode='constant') * (41 * 41)
        gap = np.zeros_like(u)
        if semantic_seen_at is not None:
            gap = ndimage.uniform_filter(
                ((np.asarray(semantic_seen_at) == 0) & (occ < -0.6)).astype(np.float32),
                size=15, mode='constant') * 225.0
        # Long-range exploration credit for large untouched areas, without
        # inventing a target behind the unknown boundary.
        score = (18.0 * np.sqrt(near_mass[iy, ix])
                 + 8.0 * np.sqrt(far_mass[iy, ix])
                 + 0.9 * np.sqrt(gap[iy, ix])
                 - 0.13 * route
                 + 0.6 * np.minimum(clr[iy, ix], 65.0))
        rx = ix // self.region_cells
        ry = iy // self.region_cells
        for n, (y, x) in enumerate(zip(ry, rx)):
            region = (int(y), int(x))
            if self.cooldowns.get(region, -1) > int(now):
                score[n] = -np.inf
                continue
            for old, ts, gain in self.history:
                if old == region and now - ts < 800:
                    score[n] -= 105.0 if gain <= 0 else 35.0
        xw = float(map_origin_xy[0]) + (ix + .5) * self.cell_px
        yw = float(map_origin_xy[1]) + (iy + .5) * self.cell_px
        for p in peer_goal_xy:
            ds = np.hypot(xw - float(p[0]), yw - float(p[1]))
            score -= 150.0 * np.clip(1.0 - ds / 160.0, 0, 1)
        for p in exclude_near_xy:
            ds = np.hypot(xw - float(p[0]), yw - float(p[1]))
            score -= 400.0 * np.clip(1.0 - ds / 120.0, 0, 1)

        # At most ONE best candidate per coarse region prevents hundreds of
        # nearly identical goals.  Selection considers *every* reachable region.
        picks: dict[tuple[int, int], int] = {}
        for i in range(len(iy)):
            if not np.isfinite(score[i]):
                continue
            region = (int(ry[i]), int(rx[i]))
            if region not in picks or score[i] > score[picks[region]]:
                picks[region] = i
        if not picks:
            return None
        selected = sorted(picks.values(), key=lambda k: (-score[k], route[k], int(iy[k]), int(ix[k])))
        best = selected[0]
        target=(int(iy[best]),int(ix[best]))
        rev=[target]
        while rev[-1]!=start:
            if rev[-1] not in parents:
                return None
            rev.append(parents[rev[-1]])
        rev.reverse()
        path=(tuple(map(float,current_xy)),)+tuple(world(c) for c in rev)
        return MacroGoal(target,(float(xw[best]),float(yw[best])),
                         (int(ry[best]),int(rx[best])),
                         float(score[best]),float(route[best]),
                         float(near_mass[target]),float(far_mass[target]),
                         path,expanded)
