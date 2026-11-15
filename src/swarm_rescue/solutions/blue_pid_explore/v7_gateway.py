"""Reachable-known-free gateway exploration for a partially known grid.

A gateway is a FREE cell adjacent to UNKNOWN cells, never a target placed
through a wall. Only a *short* LiDAR-verified probe may enter unknown space.
No competition ground truth, static map, or bomb coordinates are accessed.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Callable
import math
import heapq
import numpy as np
from scipy import ndimage

CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS = CARDINAL + ((1,1),(1,-1),(-1,1),(-1,-1))


@dataclass(frozen=True)
class Gateway:
    cell: tuple[int, int]
    xy: tuple[float, float]
    heading: float
    route: tuple[tuple[float, float], ...]
    route_cost_px: float
    unknown_gain: int
    semantic_gap: int
    score: float


@dataclass(frozen=True)
class Probe:
    xy: tuple[float, float]
    heading: float
    length_px: float
    reason: str


class GatewayPlanner:
    """Bounded eight-neighbor search with anti-corner-cutting semantics.

    The scan free-mask is a *belief*. New routes must still be validated by
    the existing hull/LiDAR machinery. Use on an event or at low frequency.
    """
    def __init__(self, origin_xy: tuple[float,float], cell_px: float = 10.0,
                 min_clearance_px: float = 16.0, max_expansions: int = 12000):
        self.origin = tuple(map(float, origin_xy))
        self.cell_px = float(cell_px)
        self.clearance = float(min_clearance_px)
        self.max_expansions = int(max_expansions)
        self.failed_until: dict[tuple[int,int],int] = {}
        self.last_selected: tuple[int,int] | None = None

    def cell(self, pos, shape):
        x = math.floor((float(pos[0])-self.origin[0])/self.cell_px)
        y = math.floor((float(pos[1])-self.origin[1])/self.cell_px)
        return (y, x) if 0<=y<shape[0] and 0<=x<shape[1] else None

    def world(self, cell):
        y, x = cell
        return (self.origin[0]+(x+.5)*self.cell_px,
                self.origin[1]+(y+.5)*self.cell_px)

    def mark_failed(self, cell: tuple[int,int], now: int, *, cooldown=350):
        self.failed_until[tuple(cell)] = int(now)+int(cooldown)

    def choose(self, *, occ: np.ndarray, clearance_px: np.ndarray,
               current_xy, now: int, semantic_seen_at: np.ndarray | None = None,
               own_evidence: np.ndarray | None = None, start_link_safe: Callable | None = None,
               peer_goals=(), max_route_px: float = 2000.0) -> Gateway | None:
        occ=np.asarray(occ,float); clear=np.asarray(clearance_px,float)
        if occ.ndim!=2 or clear.shape!=occ.shape:
            raise ValueError('occupancy and clearance must be same 2D shape')
        if semantic_seen_at is not None and np.asarray(semantic_seen_at).shape!=occ.shape:
            raise ValueError('semantic grid shape mismatch')
        if own_evidence is not None and np.asarray(own_evidence).shape!=occ.shape:
            raise ValueError('own evidence shape mismatch')
        free=(occ < -0.6) & (clear >= self.clearance) & np.isfinite(clear)
        # If local direct FREE and fused OBSTACLE disagree, schedule re-observation
        # but NEVER promote the conflicting cell to traversable.
        unknown=(occ >= -0.6) & (occ <= 1.0)
        if own_evidence is not None:
            # Direct free evidence contradicting fused occupancy is a request
            # for local verification, never permission to drive through it.
            unknown |= (np.asarray(own_evidence) < -0.8) & (occ >= -0.6)
        if not np.any(free) or not np.any(unknown):
            return None
        here=self.cell(current_xy,occ.shape)
        if here is None:
            return None
        starts=[]
        for radius in range(0, 5):
            for y in range(max(0,here[0]-radius), min(occ.shape[0],here[0]+radius+1)):
                for x in range(max(0,here[1]-radius), min(occ.shape[1],here[1]+radius+1)):
                    if free[y,x]:
                        dest=self.world((y,x))
                        if start_link_safe is not None and not start_link_safe(tuple(current_xy),dest):
                            continue
                        if start_link_safe is None and (y,x)!=here:
                            continue  # never project across a wall without local proof
                        starts.append((math.dist(current_xy,dest),(y,x)))
            if starts:
                break
        if not starts:
            return None
        start=min(starts)[1]
        # Cardinal adjacency to UNKNOWN: gateway stays known-free, even if
        # its neighbor is still unknown or has contradicted map evidence.
        adjacent_unknown=np.zeros_like(unknown)
        adjacent_unknown[1:,:] |= unknown[:-1,:]
        adjacent_unknown[:-1,:] |= unknown[1:,:]
        adjacent_unknown[:,1:] |= unknown[:,:-1]
        adjacent_unknown[:,:-1] |= unknown[:,1:]
        frontier=adjacent_unknown & free
        if not np.any(frontier):
            return None
        # A* to the NEAREST available gateway, instead of a full Dijkstra
        # over tens of thousands of known-free cells. On a 200x200 map the
        # latter previously exhausted the expansion cap before reaching a
        # remote frontier. EDT is an admissible lower bound on free-grid path.
        h_grid=ndimage.distance_transform_edt(~frontier)*self.cell_px
        dist={start:0.0}; parents={}; heap=[(float(h_grid[start]),0.,start)]
        visited=set(); found=[]; first_hit_cost=float('inf')
        while heap and len(visited)<self.max_expansions:
            _,g,c=heapq.heappop(heap)
            if c in visited:
                continue
            visited.add(c)
            if frontier[c] and self.failed_until.get(c,-1)<=now:
                found.append(c)
                if not math.isfinite(first_hit_cost):
                    first_hit_cost=g
                if len(found)>=12 or g>first_hit_cost+120.:
                    break
            for dy,dx in NEIGHBORS:
                n=(c[0]+dy,c[1]+dx)
                if not (0<=n[0]<occ.shape[0] and 0<=n[1]<occ.shape[1] and free[n]):
                    continue
                if dx and dy and (not free[c[0],n[1]] or not free[n[0],c[1]]):
                    continue
                length=self.cell_px*math.hypot(dx,dy)
                alt=g+length
                if alt<dist.get(n,float('inf')) and alt<=max_route_px:
                    dist[n]=alt;parents[n]=c
                    heapq.heappush(heap,(alt+float(h_grid[n]),alt,n))
        candidates=found
        if not candidates:
            return None
        # Small local novelty estimate; it is not 'bomb absence' proof.
        mass=ndimage.uniform_filter(unknown.astype(np.float32),size=7,mode='constant')*49
        semgap=np.zeros_like(occ,dtype=np.int32) if semantic_seen_at is None else (np.asarray(semantic_seen_at)==0).astype(np.int32)
        best=None
        for c in candidates:
            unknown_gain=int(round(float(mass[c])))
            semantic_gap=int(semgap[c])
            pt=self.world(c)
            peer_penalty=sum(32.0*max(0.0,1.0-math.dist(pt,p)/120.0) for p in peer_goals)
            score=7.0*unknown_gain+5.0*semantic_gap-0.08*dist[c]-peer_penalty
            if c==self.last_selected:
                score-=40.0
            # Vector toward adjacent unknown, no inferred connectivity beyond.
            dirs=[]
            for dy,dx in CARDINAL:
                n=(c[0]+dy,c[1]+dx)
                if 0<=n[0]<occ.shape[0] and 0<=n[1]<occ.shape[1] and unknown[n]:
                    dirs.append((dx,dy))
            vx=sum(v[0] for v in dirs);vy=sum(v[1] for v in dirs)
            heading=math.atan2(vy,vx) if dirs else 0.0
            if best is None or (score,-dist[c],c)>(best[0],-best[1],best[2]):
                best=(score,dist[c],c,heading,unknown_gain,semantic_gap)
        if best is None:
            return None
        score, g, goal,heading,nu,ns=best
        rev=[goal]
        while rev[-1]!=start:
            rev.append(parents[rev[-1]])
        rev.reverse()
        route=[tuple(current_xy)]+[self.world(c) for c in rev]
        self.last_selected=goal
        return Gateway(goal,self.world(goal),heading,tuple(route),g,nu,ns,score)


def short_unknown_probe(*, current_xy, heading:float, lidar_ranges,
                        lidar_angles, target_heading:float, hull_radius_px:float,
                        max_length_px:float=10.0, noise_margin_px:float=5.0,
                        current_segment_safe=None) -> Probe | None:
    """Only proposes a few pixels into unknown using *current* live LiDAR.

    This is not a full obstacle certificate. The caller must run the normal
    final safety filter and reject dynamically blocked or cargo states.
    """
    d=np.asarray(lidar_ranges,float); a=np.asarray(lidar_angles,float)
    if d.ndim!=1 or d.shape!=a.shape or d.size<5:
        return None
    rel=(target_heading-heading+math.pi)%(2*math.pi)-math.pi
    diff=(a-rel+math.pi)%(2*math.pi)-math.pi
    # At least three beams around requested sector to avoid an angular blind spot.
    idx=np.argsort(np.abs(diff))[:3]
    if len(idx)<3 or not np.all(np.isfinite(d[idx])):
        return None
    actual=float(np.min(d[idx]))
    reach=min(float(max_length_px),actual-float(hull_radius_px)-float(noise_margin_px))
    if reach<3.0:
        return None
    target=(float(current_xy[0])+reach*math.cos(target_heading),
            float(current_xy[1])+reach*math.sin(target_heading))
    if current_segment_safe is not None and not current_segment_safe(tuple(current_xy),target):
        return None
    return Probe(target,float(target_heading),reach,'LIVE_LIDAR_SHORT_STEP_ONLY')
