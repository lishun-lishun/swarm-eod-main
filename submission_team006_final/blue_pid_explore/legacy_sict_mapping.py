"""Shared occupancy grid, semantic scan coverage and grid path planning.

Every drone keeps an *own* lidar evidence grid (log-odds). Snapshots of it are
broadcast and relayed hop by hop, so each drone ends up with the latest grid of
every teammate it has (indirectly) been in radio range of. The fused map is the
sum of one grid per source: each observation is counted once, so a wall smeared
by one drone's compass error is voted down by everyone else's free rays.

A second grid stores, per cell, the last step at which a semantic ray could
have seen a bomb there (``scan``). Exploration targets unscanned cells, not
lidar frontiers: a cell can be lidar-mapped from 300 px away and still hide a
bomb between two of the 35 semantic rays.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

CELL = 10.0

import os as _os
CARRY_SOFT = float(_os.environ.get("V3_CARRY_SOFT", "46.0"))
CARRY_LANE = float(_os.environ.get("V3_LANE", "6.0"))
SQUEEZE = float(_os.environ.get("V3_SQUEEZE", "20.0"))

# Evidence per lidar observation (log-odds) and clamps.
FREE_STEP = -0.25
HIT_STEP = 1.2
OWN_MIN, OWN_MAX = -4.0, 6.0
FUSED_MIN, FUSED_MAX = -12.0, 18.0
OCCUPIED = 1.0          # fused evidence above which a cell is a wall

FREE_RANGE = 250.0      # free evidence is laid along rays up to this range
HIT_RANGE = 230.0       # compass error smears far hits; keep hits fairly close
SCAN_RANGE = 100.0      # 35 semantic rays: 18.5 px gap at 100 px < 24 px bomb
SAMPLE = 5.0
# Deep exploration: cells seen from up to NEAR_RANGE px (35 semantic rays leave
# a 12.6 px gap at 70 px, half a bomb), including the surface the ray hit, so a
# bomb that lidar mistook for a wall bump is still resolved by the semantics.
NEAR_RANGE = float(_os.environ.get("V3_NEAR_RANGE", "70.0"))


class GridMap:
    def __init__(self, size_area, own_id):
        width, height = size_area if size_area else (2000.0, 2000.0)
        self.width, self.height = float(width), float(height)
        self.nx = int(math.ceil(self.width / CELL))
        self.ny = int(math.ceil(self.height / CELL))
        self.own_id = own_id
        self.ev = np.zeros((self.ny, self.nx), np.float32)
        self.scan = np.zeros((self.ny, self.nx), np.int16)
        # source id -> (stamp, evidence int8 x10, scan int16)
        self.sources = {}
        self.occ = np.zeros((self.ny, self.nx), np.float32)
        self.scan_f = np.zeros((self.ny, self.nx), np.int16)
        self.near = np.zeros((self.ny, self.nx), np.int16)
        self.near_f = np.zeros((self.ny, self.nx), np.int16)
        self.fused_step = -100
        self.dirty = True
        xs = (np.arange(self.nx) + 0.5) * CELL - self.width / 2
        ys = (np.arange(self.ny) + 0.5) * CELL - self.height / 2
        self.X, self.Y = np.meshgrid(xs, ys)
        self.samples = np.arange(0.0, 300.0, SAMPLE)
        self.border = np.zeros((self.ny, self.nx), bool)
        self.border[0, :] = self.border[-1, :] = True
        self.border[:, 0] = self.border[:, -1] = True
        self.penalty = np.zeros((self.ny, self.nx), np.float32)
        self.traffic = np.zeros((self.ny, self.nx), np.float32)
        self.penalty_until = {}
        self.center_mask = np.zeros((self.ny, self.nx), bool)
        self.center_points = []
        self.seen_mask = np.zeros((self.ny, self.nx), bool)
        self.seen_points = []
        self.center_bad = set()       # cells where pushes failed
        self.costs = None
        self.clearance = None
        self.obstacle = None
        self.snapshot = None
        # Performance: cache point sets to skip redundant checks.
        self._seen_set = set()
        self._center_set = set()
        # Cache distance transform result keyed by occ array id.
        self._dt_cache_id = None
        self._dt_cache = None

    # ------------------------------------------------------------------ cells
    def cell(self, x, y):
        ix = int((x + self.width / 2) // CELL)
        iy = int((y + self.height / 2) // CELL)
        return (min(self.ny - 1, max(0, iy)), min(self.nx - 1, max(0, ix)))

    def world(self, iy, ix):
        return ((ix + 0.5) * CELL - self.width / 2,
                (iy + 0.5) * CELL - self.height / 2)

    def _flat(self, xs, ys):
        ix = np.floor((xs + self.width / 2) / CELL).astype(np.int64)
        iy = np.floor((ys + self.height / 2) / CELL).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy[ok] * self.nx + ix[ok]

    # -------------------------------------------------------------- sensing
    def integrate(self, pose, heading, dists, angles, dynamic, step):
        """Lay one lidar scan into the own evidence and scan grids.

        ``dynamic`` flags rays whose return is a drone or a bomb: their free
        segment is still used, their endpoint is not a wall.
        """
        x, y = pose
        d = np.clip(np.asarray(dists, float), 0.0, 300.0)
        a = heading + np.asarray(angles, float)
        ca, sa = np.cos(a), np.sin(a)
        s = self.samples[None, :]
        px = x + s * ca[:, None]
        py = y + s * sa[:, None]
        free = s < np.minimum(d - 10.0, FREE_RANGE)[:, None]
        hit = (d < 295.0) & (d < HIT_RANGE) & ~dynamic
        hx, hy = x + d[hit] * ca[hit], y + d[hit] * sa[hit]
        hit_idx = np.unique(self._flat(hx, hy))
        free_idx = np.unique(self._flat(px[free], py[free]))
        if hit_idx.size:
            free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)
        flat = self.ev.reshape(-1)
        flat[free_idx] += FREE_STEP
        flat[hit_idx] += HIT_STEP
        np.clip(self.ev, OWN_MIN, OWN_MAX, out=self.ev)
        scan = s < np.minimum(d - 2.0, SCAN_RANGE)[:, None]
        scan_idx = np.unique(self._flat(px[scan], py[scan]))
        self.scan.reshape(-1)[scan_idx] = min(32000, step + 1)
        near = s < np.minimum(d + 6.0, NEAR_RANGE)[:, None]
        near_idx = np.unique(self._flat(px[near], py[near]))
        self.near.reshape(-1)[near_idx] = min(32000, step + 1)
        self.dirty = True

    def mark_near(self, x, y, radius, step):
        """Declare the cells within ``radius`` as close-scanned (given up)."""
        iy, ix = self.cell(x, y)
        r = int(math.ceil(radius / CELL))
        y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - iy) ** 2 + (xx - ix) ** 2 <= r * r
        win = self.near[y0:y1, x0:x1]
        win[disk & (win == 0)] = min(32000, step + 1)
        self.dirty = True

    def add_seen_points(self, points):
        # Bulk dedup: filter points not yet in _seen_set, add them once.
        new = [(round(x, 1), round(y, 1)) for x, y in points
               if (round(x, 1), round(y, 1)) not in self._seen_set]
        if not new:
            return
        for p in new:
            self._seen_set.add(p)
            iy, ix = self.cell(*p)
            if not self.seen_mask[iy, ix]:
                self.seen_mask[iy, ix] = True
                self.seen_points.append(p)
                self.dirty = True

    def mark_center_bad(self, x, y, radius):
        """Forget arrival cells within ``radius`` (failed pushes there)."""
        iy, ix = self.cell(x, y)
        r = int(math.ceil(radius / CELL))
        cells = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                c = (iy + dy, ix + dx)
                if (0 <= c[0] < self.ny and 0 <= c[1] < self.nx
                        and dx * dx + dy * dy <= r * r):
                    cells.append(c)
        self.add_center_bad(cells)
        return cells

    def add_center_bad(self, cells):
        changed = False
        for c in cells:
            c = (int(c[0]), int(c[1]))
            if c in self.center_bad:
                continue
            self.center_bad.add(c)
            if 0 <= c[0] < self.ny and 0 <= c[1] < self.nx and self.center_mask[c]:
                self.center_mask[c] = False
            changed = True
        if changed:
            self.center_points = [p for p in self.center_points
                                  if self.cell(*p) not in self.center_bad]
            self.dirty = True

    def add_center_points(self, points):
        # Bulk dedup: filter points not yet in _center_set, add them once.
        new = [(round(x, 1), round(y, 1)) for x, y in points
               if (round(x, 1), round(y, 1)) not in self._center_set]
        if not new:
            return
        for p in new:
            self._center_set.add(p)
            iy, ix = self.cell(*p)
            if (iy, ix) in self.center_bad:
                continue
            if not self.center_mask[iy, ix]:
                self.center_mask[iy, ix] = True
                self.center_points.append(p)
                self.dirty = True

    # ----------------------------------------------------------- team fusion
    def make_snapshot(self, step):
        ev8 = np.clip(np.rint(self.ev * 10.0), -127, 127).astype(np.int8)
        self.snapshot = (step, ev8, self.scan.copy(), self.near.copy())
        return self.snapshot

    def merge_sources(self, sources):
        changed = False
        for src, entry in sources.items():
            if src == self.own_id:
                continue
            old = self.sources.get(src)
            if old is None or entry[0] > old[0]:
                self.sources[src] = entry
                changed = True
        if changed:
            self.dirty = True

    def all_sources(self):
        out = dict(self.sources)
        if self.snapshot is not None:
            out[self.own_id] = self.snapshot
        return out

    def fuse(self, step):
        if not self.dirty:
            return False
        occ = self.ev.copy()
        scan = self.scan.copy()
        near = self.near.copy()
        for entry in self.sources.values():
            occ += entry[1].astype(np.float32) * 0.1
            np.maximum(scan, entry[2], out=scan)
            if len(entry) > 3:
                np.maximum(near, entry[3], out=near)
        np.clip(occ, FUSED_MIN, FUSED_MAX, out=occ)
        self.occ, self.scan_f, self.near_f = occ, scan, near
        self.fused_step = step
        self.dirty = False
        # Invalidate distance transform cache when occ changes.
        self._dt_cache_id = None
        self._rebuild_costs(step)
        return True

    # ------------------------------------------------------------- planning
    def add_penalty(self, x, y, radius, until, weight=6.0):
        iy, ix = self.cell(x, y)
        r = int(math.ceil(radius / CELL))
        key = (iy, ix)
        self.penalty_until[key] = (until, r, weight)
        self._apply_penalties(None)

    def _apply_penalties(self, step):
        self.penalty[:] = 0.0
        for key, (until, r, weight) in list(self.penalty_until.items()):
            if step is not None and step > until:
                del self.penalty_until[key]
                continue
            iy, ix = key
            y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            self.penalty[y0:y1, x0:x1] += weight

    def set_traffic(self, spots):
        """Peer reservations: ``(x, y, radius, weight)`` in world coordinates."""
        self.traffic[:] = 0.0
        for x, y, radius, weight in spots:
            iy, ix = self.cell(x, y)
            r = int(math.ceil(radius / CELL))
            y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            np.maximum(self.traffic[y0:y1, x0:x1], weight, out=self.traffic[y0:y1, x0:x1])

    def _rebuild_costs(self, step):
        self._apply_penalties(step)
        obstacle = (self.occ > OCCUPIED) | self.border
        # Cache distance transform: only recompute when obstacle field changed.
        occ_id = id(self.occ)
        if self._dt_cache_id == occ_id and self._dt_cache is not None:
            clearance = self._dt_cache
        else:
            clearance = ndimage.distance_transform_edt(~obstacle) * CELL
            self._dt_cache_id, self._dt_cache = occ_id, clearance
        unknown = (self.occ <= OCCUPIED) & (self.occ > -0.6)
        self.obstacle = obstacle
        self.clearance = clearance
        self.unknown = unknown
        base = 1.0 + 0.35 * unknown + self.penalty + self.traffic
        soft = np.clip((42.0 - clearance) / 22.0, 0.0, 1.0)
        # Carriers keep further from wall ends: the bomb hangs off the hull
        # and snags on corners that the bare drone would clear.
        soft_wide = np.clip((CARRY_SOFT - clearance) / 28.0, 0.0, 1.0)
        # Narrow cells are expensive, never disconnected: noise can make a
        # real 47 px slot look 30 px wide, and a drone (or a carrier) that is
        # inside such a slot must still get a route out of it along the slot
        # instead of "the nearest open cell", which may lie behind a wall.
        squeeze = np.where(clearance < 14.0, SQUEEZE,
                           np.where(clearance < 20.0, SQUEEZE * 0.3, 0.0))
        passable = clearance >= 10.0
        self.costs = {
            "nav": np.where(passable, base + 3.0 * soft + squeeze, np.inf),
            "carry": np.where(passable, base + CARRY_LANE * soft_wide + 1.5 * squeeze, np.inf),
            "tight": np.where(passable, base + 4.0 * soft + 0.6 * squeeze, np.inf),
            # Exploration: narrow corridors (maze wings) must stay affordable,
            # otherwise the swarm keeps re-scanning the open side of the map.
            "explore": np.where(passable, base + 1.0 * soft + 0.25 * squeeze, np.inf),
        }

    def free_start(self, cost, x, y, radius=4):
        """Nearest passable cell to (x, y) (the drone may hug a wall)."""
        iy, ix = self.cell(x, y)
        if np.isfinite(cost[iy, ix]):
            return iy, ix
        best = None
        for r in range(1, radius + 1):
            y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            window = cost[y0:y1, x0:x1]
            ok = np.argwhere(np.isfinite(window))
            if ok.size:
                cand = ok + (y0, x0)
                d2 = (cand[:, 0] - iy) ** 2 + (cand[:, 1] - ix) ** 2
                best = tuple(cand[int(np.argmin(d2))])
                break
        return best

    def dijkstra(self, kind, starts):
        cost = self.costs[kind]
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs(starts)
        return mcp, cum

    def path_from(self, mcp, end):
        try:
            cells = mcp.traceback(end)
        except (ValueError, IndexError):
            return []
        return [self.world(iy, ix) for iy, ix in cells]

    def segment_clear(self, a, b, min_clear=16.0):
        """True if every sample of segment a-b keeps ``min_clear`` from walls."""
        if self.clearance is None:
            return True
        length = math.dist(a, b)
        n = max(2, int(length / 5.0))
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        ix = np.clip(((xs + self.width / 2) // CELL).astype(int), 0, self.nx - 1)
        iy = np.clip(((ys + self.height / 2) // CELL).astype(int), 0, self.ny - 1)
        return bool(np.all(self.clearance[iy, ix] >= min_clear))

    def clearance_at(self, x, y):
        if self.clearance is None:
            return 100.0
        iy, ix = self.cell(x, y)
        return float(self.clearance[iy, ix])


