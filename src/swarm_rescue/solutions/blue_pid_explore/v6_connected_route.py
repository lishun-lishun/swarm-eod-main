"""V6: conservative connected A* on legal belief, no through-wall projection.

Cost fields describe map route cost only, never physical arrival or disposal.
Unknown, conflicting and obstacle cells are NOT granted free passage.
A segment validator for the current drone/payload footprint is mandatory for
live route execution; A* alone cannot certify a swinging jointed payload.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import numpy as np


@dataclass(frozen=True)
class RouteResult:
    status: str
    points: tuple[tuple[float, float], ...]
    length_px: float
    expansions: int
    reason: str


class KnownFreeAStar:
    NEIGHBORS = ((-1,0), (1,0), (0,-1), (0,1),
                 (-1,-1), (-1,1), (1,-1), (1,1))

    def __init__(self, occ: np.ndarray, clearance_px: np.ndarray,
                 *, origin_xy=(-1000.0, -1000.0), cell_px=10.0,
                 min_clearance_px=18.0, max_expansions=10000):
        occ = np.asarray(occ, dtype=float)
        clr = np.asarray(clearance_px, dtype=float)
        if occ.ndim != 2 or occ.shape != clr.shape:
            raise ValueError('occupancy and clearance need same 2D shape')
        self.free = ((occ < -0.6) & (clr >= min_clearance_px))
        self.clearance = clr
        self.origin = tuple(float(v) for v in origin_xy)
        self.cell_px = float(cell_px)
        self.max_expansions = int(max_expansions)
        if self.cell_px <= 0 or self.max_expansions <= 0:
            raise ValueError('invalid planner parameters')
        self.shape = occ.shape

    def cell(self, xy):
        ix = int(math.floor((float(xy[0]) - self.origin[0]) / self.cell_px))
        iy = int(math.floor((float(xy[1]) - self.origin[1]) / self.cell_px))
        if ix < 0 or iy < 0 or ix >= self.shape[1] or iy >= self.shape[0]:
            return None
        return (iy, ix)

    def world(self, cell):
        y,x = cell
        return (self.origin[0]+(x+.5)*self.cell_px,
                self.origin[1]+(y+.5)*self.cell_px)

    def in_free(self, cell):
        return (cell is not None and 0 <= cell[0] < self.shape[0]
                and 0 <= cell[1] < self.shape[1] and bool(self.free[cell]))

    def start_cell(self, xy, local_segment_safe, *, radius_cells=4):
        """Nearest known-free snap only if directly verified by local sensor/hull.

        Reject an arbitrary nearest-free projection across a wall.
        """
        c = self.cell(xy)
        if c is None:
            return None
        options=[]
        for dy in range(-radius_cells,radius_cells+1):
            for dx in range(-radius_cells,radius_cells+1):
                n=(c[0]+dy,c[1]+dx)
                if self.in_free(n):
                    options.append((math.hypot(dx,dy),n))
        for _,n in sorted(options):
            if local_segment_safe(tuple(xy), self.world(n)):
                return n
        return None

    def goal_cell(self, xy, *, radius_cells=3):
        c=self.cell(xy)
        if c is None:
            return None
        options=[]
        for dy in range(-radius_cells,radius_cells+1):
            for dx in range(-radius_cells,radius_cells+1):
                n=(c[0]+dy,c[1]+dx)
                if self.in_free(n):
                    options.append((math.hypot(dx,dy),n))
        return min(options)[1] if options else None

    def plan(self, start_xy, goal_xy, *, local_segment_safe,
             goal_radius_cells=3) -> RouteResult:
        if local_segment_safe is None:
            raise ValueError('local hull-aware segment validation required')
        s = self.start_cell(start_xy, local_segment_safe)
        g = self.goal_cell(goal_xy, radius_cells=goal_radius_cells)
        if s is None:
            return RouteResult('INVALID_START', (), math.inf, 0,
                               'no verified connector to nearest free cell')
        if g is None:
            return RouteResult('INVALID_GOAL', (), math.inf, 0,
                               'goal has no known-free standable approach cell')
        dist={s:0.0}; prev={}; queue=[]; counter=0
        heuristic=lambda c:self.cell_px*math.hypot(c[0]-g[0],c[1]-g[1])
        heapq.heappush(queue,(heuristic(s),counter,s)); closed=set(); expansions=0
        while queue and expansions < self.max_expansions:
            _,_,cur=heapq.heappop(queue)
            if cur in closed:
                continue
            closed.add(cur); expansions+=1
            if cur==g:
                cells=[cur]
                while cells[-1]!=s:
                    cells.append(prev[cells[-1]])
                cells.reverse()
                pts=[tuple(start_xy)]
                pts.extend(self.world(c) for c in cells)
                return RouteResult('MAP_CONNECTED',tuple(pts),dist[cur],expansions,
                    'known-free map connection only; live hull verification still required')
            for dy,dx in self.NEIGHBORS:
                nb=(cur[0]+dy,cur[1]+dx)
                if not self.in_free(nb) or nb in closed:
                    continue
                if dx and dy and (not self.in_free((cur[0],nb[1]))
                                  or not self.in_free((nb[0],cur[1]))):
                    continue  # never diagonally cut wall corners
                # Clearance penalization, without making real narrow corridors
                # mathematically disconnected at a guessed cargo radius.
                step=self.cell_px*math.hypot(dx,dy)
                soft=max(0.0,30.0-float(self.clearance[nb]))/30.0
                new=dist[cur]+step*(1.0+soft)
                if new < dist.get(nb,math.inf):
                    dist[nb]=new;prev[nb]=cur;counter+=1
                    heapq.heappush(queue,(new+heuristic(nb),counter,nb))
        status='BUDGET' if queue else 'DISCONNECTED'
        return RouteResult(status,(),math.inf,expansions,
                           'no complete connected known-free route established')


@dataclass(frozen=True)
class TrackingResult:
    status: str
    waypoint: tuple[float,float]|None
    index: int


def verified_lookahead(current_xy, points, *, index:int,
                       segment_safe, reach_px=12.0, lookahead_px=42.0):
    """Monotone cursor; never choose nearest point from distant path segment."""
    path=tuple(tuple(p) for p in points)
    if not path:
        return TrackingResult('NO_PATH',None,max(0,index))
    here=tuple(current_xy)
    idx=max(0,min(index,len(path)-1))
    # Consume nearby sequential waypoints, retaining the last *verified* one.
    # This preserves a safe fallback if the following waypoint is blocked.
    best=None; best_idx=idx
    while idx<len(path)-1 and math.dist(here,path[idx])<=reach_px:
        if segment_safe(here,path[idx]):
            best=path[idx];best_idx=idx
        idx+=1
    for j in range(idx,len(path)):
        if math.dist(here,path[j])>lookahead_px:
            break
        if not segment_safe(here,path[j]):
            break
        best=path[j];best_idx=j
    if best is None:
        return TrackingResult('NO_VERIFIED_SEGMENT',None,idx)
    return TrackingResult('TRACK',best,best_idx)


def reversed_legal_breadcrumbs(current_xy, trace_points, *, segment_safe,
                                 max_steps=20):
    """Safe prefix of already traversed carry history, reversed for retreat."""
    p=tuple(current_xy); result=[p]
    for prev in reversed(tuple(trace_points)):
        q=tuple(prev)
        if math.dist(p,q)<7.0:
            continue
        if not segment_safe(p,q):
            break
        result.append(q);p=q
        if len(result)>=max_steps+1:
            break
    return RouteResult('VERIFIED_RETREAT' if len(result)>1 else 'NO_VERIFIED_RETREAT',
                       tuple(result),sum(math.dist(a,b) for a,b in zip(result,result[1:])),
                       0,'retreat requires the currently observed cargo footprint clearance')
