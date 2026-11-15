"""Robot-private 2D occupancy: static and moving-target evidence are separated."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import V9Config
from .geometry import Pose, RelativeDetection, wrap

class BeliefGrid:
    def __init__(self, area_xy=(2000.,2000.), config:V9Config|None=None):
        self.cfg=config or V9Config()
        self.cell_px=self.cfg.cell_px
        width,height=map(float,area_xy or (2000.,2000.))
        self.w=max(1,int(math.ceil(width/self.cell_px)))
        self.h=max(1,int(math.ceil(height/self.cell_px)))
        # Local frame origin is at initial drone pose, not necessarily map center.
        # Keeping +/- half arena available is usually enough if start is near center;
        # the adapter should expand / recenter if it approaches map edges.
        self.origin=(-width/2.,-height/2.)
        self.log_odds=np.zeros((self.h,self.w),dtype=np.float32)
        self.sem_rays=np.zeros((self.h,self.w),dtype=np.uint16)
        self.last_seen=np.zeros((self.h,self.w),dtype=np.int32)
        self.occ_threshold=self.cfg.map_occ_threshold
        self.free_threshold=self.cfg.map_free_threshold
        self.revision=0
        self._cached_clear=None
        self._cached_revision=-1

    def cell(self,xy):
        x=int(math.floor((float(xy[0])-self.origin[0])/self.cell_px))
        y=int(math.floor((float(xy[1])-self.origin[1])/self.cell_px))
        return (y,x) if 0<=x<self.w and 0<=y<self.h else None

    def xy(self,c):
        y,x=c
        return (self.origin[0]+(x+.5)*self.cell_px,
                self.origin[1]+(y+.5)*self.cell_px)

    def trace(self,a,b):
        """Bresenham integer cells; bounded by the caller's sensor range."""
        c0,c1=self.cell(a),self.cell(b)
        if c0 is None:
            return []
        if c1 is None:
            # Clip a sensor endpoint outside the allocated LOCAL map, so rays
            # near the border still contribute all traversed interior cells.
            xx=min(max(float(b[0]),self.origin[0]+.1),self.origin[0]+self.w*self.cell_px-.1)
            yy=min(max(float(b[1]),self.origin[1]+.1),self.origin[1]+self.h*self.cell_px-.1)
            c1=self.cell((xx,yy))
            if c1 is None:return []
        y0,x0=c0; y1,x1=c1
        dx=abs(x1-x0);dy=abs(y1-y0)
        sx=1 if x0<x1 else -1;sy=1 if y0<y1 else -1
        err=dx-dy; path=[]
        while True:
            path.append((y0,x0))
            if (x0,y0)==(x1,y1):break
            e2=2*err
            if e2>-dy:err-=dy;x0+=sx
            if e2<dx:err+=dx;y0+=sy
            if len(path)>max(self.w,self.h)*2:break
        return path

    def integrate_lidar(self,pose:Pose,ranges,angles,step:int,
                        dynamic_detections=()):
        rr=np.asarray(ranges,dtype=float); aa=np.asarray(angles,dtype=float)
        if rr.ndim!=1 or rr.shape!=aa.shape:return 0
        before=int(np.count_nonzero(self.log_odds<self.free_threshold))
        for d,a in zip(rr,aa):
            if not math.isfinite(float(d)) or d<=0:continue
            hit=bool(d<self.cfg.lidar_range_px-3.0)
            extent=min(float(d),self.cfg.lidar_range_px-1)
            endpoint_dynamic=False
            for det in dynamic_detections:
                if det.kind not in ('bomb','drone'):continue
                half=14.0 if det.kind=='bomb' else 20.0
                angular_width=math.atan2(half+6.0,max(1.,det.distance))+.04
                if (abs(wrap(float(a)-det.angle))<angular_width
                        and (det.distance-24.)<float(d)<(det.distance+46.)):
                    endpoint_dynamic=True;break
            # Dynamic endpoint is not a permanent wall.
            length=max(0.,extent-(4. if hit else 0.))
            e=(pose.x+length*math.cos(pose.yaw+a),
               pose.y+length*math.sin(pose.yaw+a))
            route=self.trace((pose.x,pose.y),e)
            if not route:continue
            for c in route[:-1]:
                self.log_odds[c]=max(-4.0,float(self.log_odds[c])-0.17)
                self.last_seen[c]=step
            end=route[-1]
            if hit and not endpoint_dynamic:
                self.log_odds[end]=min(4.0,float(self.log_odds[end])+0.48)
            else:
                self.log_odds[end]=max(-4.0,float(self.log_odds[end])-0.14)
            self.last_seen[end]=step
        self.revision+=1
        self._cached_clear=None
        return int(np.count_nonzero(self.log_odds<self.free_threshold))-before

    def integrate_semantic_geometry(self,pose:Pose,lidar_ranges,lidar_angles,
                                    step:int,range_px:float=130.0):
        """Ray passage counts, NOT evidence that there is no bomb."""
        rr=np.asarray(lidar_ranges,dtype=float)
        aa=np.asarray(lidar_angles,dtype=float)
        if rr.shape!=aa.shape or rr.ndim!=1 or rr.size<40:return 0
        # Official angles use inclusive 35 linspace, endpoints duplicate.
        angles=np.linspace(-math.pi,math.pi,35)
        count=0
        for a in angles:
            delta=(aa-a+math.pi)%(2*math.pi)-math.pi
            j=int(np.argmin(abs(delta)))
            d=rr[j]
            if not math.isfinite(d):continue
            length=max(0.,min(range_px,float(d)-5.))
            end=(pose.x+length*math.cos(pose.yaw+a),
                 pose.y+length*math.sin(pose.yaw+a))
            for c in self.trace((pose.x,pose.y),end):
                if self.sem_rays[c]==0:count+=1
                self.sem_rays[c]=min(65535,int(self.sem_rays[c])+1)
        return count

    def obstacle_mask(self):
        mask=self.log_odds>self.occ_threshold
        mask=np.asarray(mask, dtype=bool)
        mask[[0,-1],:]=True; mask[:,[0,-1]]=True
        return mask

    def free_mask(self,clearance_px=0.0):
        f=self.log_odds<self.free_threshold
        if clearance_px>0:
            f &= self.clearance()>=clearance_px
        f[[0,-1],:]=False; f[:,[0,-1]]=False
        return f

    def clearance(self):
        if self._cached_revision!=self.revision or self._cached_clear is None:
            # CRITICAL: clearance is distance to KNOWN OCCUPIED cells only.
            # Unknown cells are disallowed by free_mask() itself, but they are
            # NOT walls. Treating UNKNOWN as an inflated wall makes frontier
            # viewpoints adjacent to unknown impossible to reach.
            blocking=self.log_odds>self.occ_threshold
            blocking[[0,-1],:]=True;blocking[:,[0,-1]]=True
            self._cached_clear=distance_transform_edt(~blocking)*self.cell_px
            self._cached_revision=self.revision
        return self._cached_clear

    def known_free_segment(self,start,end,*,min_clear_px=15.):
        cells=self.trace(start,end)
        if not cells:return False
        free=self.free_mask(clearance_px=min_clear_px)
        return all(bool(free[c]) for c in cells)

    def ray_free_distance(self,pose:Pose,ranges,angles,direction_local,
                          *,half_angle=.12,quantile=.18):
        rr=np.asarray(ranges,dtype=float);aa=np.asarray(angles,dtype=float)
        if rr.shape!=aa.shape or rr.ndim!=1:return 0.
        diff=np.abs((aa-direction_local+math.pi)%(2*math.pi)-math.pi)
        selected=rr[(diff<=half_angle)&np.isfinite(rr)&(rr>0)]
        return float(np.quantile(selected,quantile)) if selected.size else 0.
