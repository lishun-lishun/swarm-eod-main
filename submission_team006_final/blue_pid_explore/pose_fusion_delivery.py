"""Witnessed delivery adapter for V2; based on eod/delivery.py's evidence model.

Uses only estimates, semantic surface observations and the V2 grid. Failed
approaches expire; unchecked radial goals and relaxed failed goals are excluded.
"""
import math

class ApproachSet:
    def __init__(self):
        self.witnesses={}; self.penalties=[]; self.version=0

    def observe(self,grid,pose,hits,step):
        for x,y,d in hits[::max(1,len(hits)//5)]:
            length=math.dist(pose,(x,y))
            if length<1: continue
            # Preserve the actual observer as well as its map-checked stand-off.
            points=[pose]
            t=min(40.,length)/length
            stand=(x+t*(pose[0]-x),y+t*(pose[1]-y))
            if grid.segment_clear(pose,stand,16.): points.append(stand)
            for p in points:
                key=grid.cell(*p); value=(float(p[0]),float(p[1]),float(x),float(y),int(step))
                if key not in self.witnesses: self.version+=1
                self.witnesses[key]=value
        if len(self.witnesses)>120:
            self.witnesses=dict(sorted(self.witnesses.items(),key=lambda kv:-kv[1][4])[:96])

    def merge(self,grid,items,step):
        if not isinstance(items,(tuple,list)): return
        for w in items[:96]:
            if not isinstance(w,(tuple,list)) or len(w)!=5: continue
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in w): continue
            if not (0<=w[4]<=step and abs(w[0])<grid.width/2 and abs(w[1])<grid.height/2): continue
            k=grid.cell(*w[:2]); old=self.witnesses.get(k)
            if old is None or w[4]>old[4]:
                self.witnesses[k]=tuple(w)
                if old is None: self.version+=1
        if len(self.witnesses)>120: self.witnesses=dict(sorted(self.witnesses.items(),key=lambda kv:-kv[1][4])[:96])

    def blocked(self,p,step):
        return any(until>step and math.dist(p,q)<radius for q,radius,until in self.penalties)

    def fail(self,p,step):
        level=sum(math.dist(p,q)<50 for q,_,until in self.penalties if until>step-600)
        self.penalties.append((tuple(p),45.,step+min(1800,450*(level+1))))
        self.penalties=self.penalties[-24:]; self.version+=1

    def valid(self,grid,step):
        if grid.costs is None: return []
        out=[]
        for w in self.witnesses.values():
            p=w[:2]; surface=w[2:4]; cell=grid.cell(*p)
            if self.blocked(p,step) or not math.isfinite(grid.costs['carry'][cell]): continue
            d=math.dist(p,surface)
            if d>130 or grid.clearance_at(*p)<12: continue
            t=max(0.,d-22.)/max(d,1.)
            end=(p[0]+t*(surface[0]-p[0]),p[1]+t*(surface[1]-p[1]))
            if grid.segment_clear(p,end,12.): out.append(w)
        return out

    def payload(self):
        return tuple(sorted(self.witnesses.values(),key=lambda w:-w[4])[:64])


