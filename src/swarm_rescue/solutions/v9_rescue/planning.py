"""Topology-first navigation: known-free A*, global reachable frontier search,
short LiDAR-only probing at explored/unknown boundary, and local breadcrumbs.
"""
from __future__ import annotations
import heapq
import math
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter, binary_dilation
from .geometry import Pose, wrap, normalize
from .mapping import BeliefGrid

CARD=((1,0),(-1,0),(0,1),(0,-1))
EIGHT=CARD+((1,1),(1,-1),(-1,1),(-1,-1))

@dataclass(frozen=True)
class PathPlan:
    status: str
    points: tuple[tuple[float,float],...]
    distance: float
    expansions: int
    detail: str=''

@dataclass(frozen=True)
class FrontierGoal:
    world: tuple[float,float]  # ego-local coordinates, NOT simulator world
    route: PathPlan
    utility: float
    estimated_unknown: float
    region: tuple[int,int]
    into_unknown_yaw: float

class KnownFreePlanner:
    def __init__(self, belief:BeliefGrid):
        self.grid=belief
        self.last_revision=-1
        self.frontier_cooldown={}
        self.history=deque(maxlen=50)
        self._distance_cache_key=None
        self._distance_cache=None

    def _nearest_free(self,p,free,*,local_link=None,radius=5):
        c=self.grid.cell(p)
        if c is None:return None
        options=[]
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                n=(c[0]+dy,c[1]+dx)
                if not (0<=n[0]<self.grid.h and 0<=n[1]<self.grid.w and free[n]):continue
                dest=self.grid.xy(n)
                options.append((math.dist(p,dest),n))
        if local_link is None:
            return c if c is not None and free[c] else None
        for _,candidate in sorted(options):
            if local_link(p,self.grid.xy(candidate)):
                return candidate
        return None

    def _dijkstra(self,start,free,*,max_expansions=26000):
        dist={start:0.};prev={};queue=[(0.,start)];count=0
        while queue and count<max_expansions:
            d,c=heapq.heappop(queue)
            if d>dist.get(c,math.inf)+1e-7:continue
            count+=1
            for dy,dx in EIGHT:
                n=(c[0]+dy,c[1]+dx)
                if not (0<=n[0]<self.grid.h and 0<=n[1]<self.grid.w and free[n]):continue
                if dx and dy and (not free[c[0],n[1]] or not free[n[0],c[1]]):continue
                nd=d+self.grid.cell_px*math.hypot(dx,dy)
                if nd<dist.get(n,math.inf):
                    dist[n]=nd;prev[n]=c;heapq.heappush(queue,(nd,n))
        return dist,prev,count

    def distances_from(self,start_xy,*,local_link=None,clearance_px=18.):
        """One distance field for all task bids, not one A* per task."""
        free=self.grid.free_mask(clearance_px)
        c=self._nearest_free(start_xy,free,local_link=local_link,radius=4)
        if c is None:return {},0
        key=(c,int(self.grid.revision)//5,round(float(clearance_px),1))
        if self._distance_cache_key!=key:
            dist,_,expanded=self._dijkstra(c,free,max_expansions=26000)
            self._distance_cache_key=key
            self._distance_cache=(dist,expanded)
        return self._distance_cache

    def path(self,start_xy,target_xy,*,carrying=False,local_link=None):
        clearance=18.0 if not carrying else 25.0
        free=self.grid.free_mask(clearance)
        start=self._nearest_free(start_xy,free,local_link=local_link,radius=5)
        if start is None:return PathPlan('NO_START',(),math.inf,0)
        target=self.grid.cell(target_xy)
        if target is None:return PathPlan('OUTSIDE',(),math.inf,0)
        reachable,prev,count=self._dijkstra(start,free)
        near=[]
        for dy in range(-5,6):
            for dx in range(-5,6):
                c=(target[0]+dy,target[1]+dx)
                if c in reachable:near.append((math.dist(target_xy,self.grid.xy(c)),c))
        if not near:return PathPlan('NOT_CONNECTED',(),math.inf,count)
        terminal=min(near)[1]
        cells=[terminal]
        while cells[-1]!=start:
            cells.append(prev[cells[-1]])
        cells.reverse()
        route=[tuple(float(v) for v in start_xy)]
        route += [self.grid.xy(c) for c in cells]
        return PathPlan('CONNECTED',tuple(route),float(reachable[terminal]),count)

    def select_frontier(self,current_xy,*,now:int,drone_id:int,
                        peer_destinations=(),local_link=None,
                        max_expansions=26000):
        grid=self.grid
        free=grid.free_mask(clearance_px=17.0)
        start=self._nearest_free(current_xy,free,local_link=local_link,radius=5)
        if start is None:return None
        distances,previous,expansions=self._dijkstra(start,free,max_expansions=max_expansions)
        unknown=(grid.log_odds>=grid.free_threshold)&(grid.log_odds<=grid.occ_threshold)
        adjacent=np.zeros_like(unknown)
        adjacent[1:,:] |= unknown[:-1,:]
        adjacent[:-1,:] |= unknown[1:,:]
        adjacent[:,1:] |= unknown[:,:-1]
        adjacent[:,:-1] |= unknown[:,1:]
        candidates=np.argwhere(adjacent & free)
        if len(candidates)==0:return None
        # Coarse information gain is scored only on reachable candidates.
        u_near=uniform_filter(unknown.astype(float),size=17)*289.0
        u_far=uniform_filter(unknown.astype(float),size=41)*1681.0
        sem_gap=uniform_filter((grid.sem_rays==0).astype(float),size=15)*225.0
        best=None;weights=(.9,.35,.18)
        # Evaluate all connected frontier regions, not just nearest 12.
        for cy,cx in candidates:
            c=(int(cy),int(cx))
            d=distances.get(c)
            if d is None or d>3300:continue
            region=(c[0]//20,c[1]//20)
            if self.frontier_cooldown.get(region,-1)>now:continue
            pt=grid.xy(c)
            peer_cost=sum(90*max(0.,1.-math.dist(pt,p)/220.) for p in peer_destinations)
            # Deterministic soft sector diversity: tie preference, never a wall.
            a=math.atan2(pt[1]-current_xy[1],pt[0]-current_xy[0])
            nominal=-math.pi+2*math.pi*((drone_id*0.61803398875)%1)
            sector=28*math.cos(wrap(a-nominal))
            novelty=(weights[0]*math.sqrt(max(0.,u_near[c]))+
                     weights[1]*math.sqrt(max(0.,u_far[c]))+
                     weights[2]*math.sqrt(max(0.,sem_gap[c])))
            revisit=sum(40*max(0.,1.-math.dist(pt,hp)/130.) for hp in self.history)
            utility=novelty-.048*d-peer_cost-revisit+sector
            if best is None or utility>best[0]:best=(utility,c,d,region,novelty)
        if best is None:return None
        utility,cell,distance,region,gain=best
        cells=[cell]
        while cells[-1]!=start:
            if cells[-1] not in previous:return None
            cells.append(previous[cells[-1]])
        cells.reverse()
        points=[tuple(float(v) for v in current_xy)]+[grid.xy(c) for c in cells]
        # The probe direction must point into actual neighboring unknown cells,
        # not back toward the frontier waypoint that is already reached.
        neighbor_dirs=[]
        for dy,dx in CARD:
            n=(cell[0]+dy,cell[1]+dx)
            if 0<=n[0]<grid.h and 0<=n[1]<grid.w and unknown[n]:
                neighbor_dirs.append((dx,dy))
        vx=sum(d[0] for d in neighbor_dirs);vy=sum(d[1] for d in neighbor_dirs)
        yaw=math.atan2(vy,vx) if neighbor_dirs else 0.0
        return FrontierGoal(grid.xy(cell),PathPlan('CONNECTED',tuple(points),distance,expansions),utility,gain,region,yaw)

    def on_frontier_finished(self,goal:FrontierGoal,now:int,*,gained_cells:int,arrived:bool):
        self.history.append(goal.world)
        self.frontier_cooldown[goal.region]=now+(500 if not arrived or gained_cells<=0 else 130)

class RouteFollower:
    def __init__(self):
        self.route=PathPlan('NONE',(),math.inf,0)
        self.index=0
        self.last_goal=None
        self.last_progress_step=-1
        self.best_distance=math.inf
        self.hold=False

    def set_route(self,route:PathPlan,now:int):
        if route.status!='CONNECTED' or len(route.points)<2:return False
        self.route=route;self.index=1
        self.last_progress_step=now;self.best_distance=math.inf
        return True

    def reset(self):
        self.__init__()

    def waypoint(self,pos,*,now:int,local_segment_ok=None,
                 lookahead=48.,tolerance=10.):
        path=self.route.points
        if not path:return None
        while self.index<len(path)-1 and math.dist(pos,path[self.index])<=tolerance:
            self.index+=1
        target=path[self.index]
        # Only skip waypoints if CURRENT direct segment is validated.
        for j in range(self.index+1,len(path)):
            if math.dist(pos,path[j])>lookahead:break
            if local_segment_ok is None or not local_segment_ok(pos,path[j]):break
            target=path[j];self.index=j
        distance=math.dist(pos,target)
        if distance<self.best_distance-2.5:
            self.best_distance=distance;self.last_progress_step=now
        return target

    def stalled(self,now:int,timeout=90):
        return self.last_progress_step>=0 and now-self.last_progress_step>timeout

class EgoBreadcrumbs:
    """Records all legal ego-local traversal from birth, not only after pickup.
    Backtracking an empty route with cargo is NOT certified by history alone.
    """
    def __init__(self,max_points=1000):
        self.points=deque(maxlen=max_points)
        self.last_step=-1

    def add(self,pos,now:int,*,min_interval=8,min_distance=7.):
        p=tuple(float(v) for v in pos)
        if (not self.points or math.dist(self.points[-1][1],p)>=min_distance
                or now-self.last_step>=min_interval):
            self.points.append((int(now),p));self.last_step=int(now)

    def local_backtrack(self,pos,*,current_safe,max_segment=24.,min_gap=6.,max_age=200):
        if not self.points:return None
        latest_time=self.points[-1][0]
        for step,p in reversed(self.points):
            d=math.dist(pos,p)
            if latest_time-step>max_age:break
            if min_gap<d<=max_segment and current_safe(pos,p):return p
        return None


def lidar_unknown_step(pose:Pose,ranges,angles,*,body_radius_px:float,
                       yaw_relative:float,max_step_px=8.,noise_margin=7.):
    """Only proposes a small forward step when current obstacle rays support it.
    Does not certify unknown map space or carried load; final safety still needed.
    """
    d=np.asarray(ranges,dtype=float);a=np.asarray(angles,dtype=float)
    if d.ndim!=1 or d.shape!=a.shape or d.size<20:return None
    err=np.abs((a-yaw_relative+math.pi)%(2*math.pi)-math.pi)
    ids=np.argsort(err)[:5]
    if np.any(~np.isfinite(d[ids])):return None
    clearance=float(np.min(d[ids]))-body_radius_px-noise_margin
    step=min(max_step_px,clearance)
    if step<3.:return None
    heading=pose.yaw+yaw_relative
    return (pose.x+step*math.cos(heading),pose.y+step*math.sin(heading))
