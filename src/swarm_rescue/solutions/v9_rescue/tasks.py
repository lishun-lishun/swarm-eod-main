"""Event-driven one-task scheduling with explicit limited-radio uncertainty.

Not a full CBBA implementation: no fake global consensus across disconnections.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
import numpy as np
from .geometry import RelativeDetection, Pose

@dataclass
class Target:
    key: str
    xy: np.ndarray
    last_seen: int
    local_observed: bool
    state: str = 'KNOWN'  # KNOWN, CARRIED, DISPOSED, SUSPECT_DROP
    attempts: int = 0
    view_xy: np.ndarray|None = None

@dataclass
class Peer:
    uid: int
    last_rx: int
    world_xy: tuple[float,float]|None
    target_world: tuple[float,float]|None
    carrying: bool
    phase: str
    raw: dict=field(default_factory=dict)

class TaskMemory:
    def __init__(self, own_id:int, match_px=24.0):
        self.own_id=int(own_id)
        self.match=float(match_px)
        self.targets:dict[str,Target]={}
        self.next_key=0
        self.peers:dict[int,Peer]={}
        self.my_target:str|None=None
        self.center_local:np.ndarray|None=None
        self.center_step=-1
        self.latest_grasp_key=None
        self.disposed_seen=0
        self._last_claim_change=-100000
        self._last_pickup_step=-100000
        self._last_grasped=False
        self._last_reward=None
        self.events=[]

    def log(self,step,kind,**kw):
        self.events.append({'step':int(step),'kind':str(kind),**kw})
        if len(self.events)>160:self.events=self.events[-160:]

    def closest(self,xy,*,allow_carried=False):
        p=np.asarray(xy,dtype=float)
        best=None;dist=self.match
        for t in self.targets.values():
            if t.state=='DISPOSED' or (t.state=='CARRIED' and not allow_carried):continue
            d=float(np.linalg.norm(t.xy-p))
            if d<dist:best=t;dist=d
        return best

    def observe_bomb(self,xy,step:int,view_xy=None):
        p=np.asarray(xy,dtype=float).reshape(2)
        if not np.all(np.isfinite(p)):return None
        target=self.closest(p)
        if target is None:
            key=f'{self.own_id}:{self.next_key}';self.next_key+=1
            target=Target(key,p.copy(),int(step),True,view_xy=None if view_xy is None else np.array(view_xy,dtype=float))
            self.targets[key]=target
            self.log(step,'FIRST_SEEN',key=key)
        else:
            target.xy=.8*target.xy+.2*p
            target.last_seen=int(step)
            target.local_observed=True
            if view_xy is not None:target.view_xy=np.array(view_xy,dtype=float)
        return target

    def observe_center(self,surface_xy,step):
        p=np.asarray(surface_xy,dtype=float).reshape(2)
        if np.isfinite(p).all():
            # Approximate surface only. Never silently call this center position.
            self.center_local=p if self.center_local is None else .85*self.center_local+.15*p
            self.center_step=int(step)

    def read_messages(self,messages,now:int,localizer):
        """Accept only messages from official communicator.received_messages."""
        for message in messages or ():
            # Official communicator yields (sender_id, payload).
            if not isinstance(message,(tuple,list)) or len(message)<2:continue
            data=message[1]
            if not isinstance(data,dict) or data.get('schema')!='v9.1':continue
            try:uid=int(data['id'])
            except (ValueError,TypeError,KeyError):continue
            if uid==self.own_id:continue
            world_pos=data.get('world_xy')
            wtarget=data.get('task_world')
            world_pos=tuple(map(float,world_pos)) if isinstance(world_pos,(list,tuple)) and len(world_pos)==2 else None
            wtarget=tuple(map(float,wtarget)) if isinstance(wtarget,(list,tuple)) and len(wtarget)==2 else None
            self.peers[uid]=Peer(uid,now,world_pos,wtarget,bool(data.get('carrying')),str(data.get('phase','UNKNOWN')),data)
            if not localizer.anchored:continue
            # Do not merge maps / GPS-denied pose claims into the local grid.
            if data.get('pose_anchored') is not True:continue
            for entry in data.get('targets',())[:24]:
                if not isinstance(entry,dict) or entry.get('state') not in ('KNOWN','CARRIED','DISPOSED'):continue
                world=entry.get('world')
                if not isinstance(world,(list,tuple)) or len(world)!=2:continue
                q=localizer.local_from_world(world)
                if q is None:continue
                # Import only into comparable anchored coordinate frame.
                t=min((target for target in self.targets.values()
                       if np.linalg.norm(target.xy-q)<min(self.match,14.)),
                      key=lambda target:np.linalg.norm(target.xy-q),default=None)
                state=entry['state']
                if t is not None and state in ('CARRIED','DISPOSED'):
                    if t.state!=state and t.state!='DISPOSED':
                        t.state=state
                        self.log(now,'PEER_'+state,key=t.key,peer=uid)
                        if self.my_target==t.key:self.my_target=None
                elif t is None and state=='KNOWN' and np.isfinite(q).all():
                    k=f'{uid}:remote:{self.next_key}';self.next_key+=1
                    self.targets[k]=Target(k,q.copy(),now,False)
            world_center=data.get('center_surface_world')
            if self.center_local is None and isinstance(world_center,(list,tuple)) and len(world_center)==2:
                cp=localizer.local_from_world(world_center)
                if cp is not None and np.isfinite(cp).all():self.center_local=np.asarray(cp)
        self.peers={pid:p for pid,p in self.peers.items() if now-p.last_rx<=35}

    def observe_grasp_transition(self,step:int,carrying:bool,reward=None,at_xy=None):
        carrying=bool(carrying)
        if carrying and not self._last_grasped:
            self._last_pickup_step=int(step)
            key=self.my_target
            if key in self.targets:
                self.targets[key].state='CARRIED'
                self.latest_grasp_key=key
            self.log(step,'GRASP_CONFIRMED',key=key)
            self.my_target=None
        if not carrying and self._last_grasped:
            confirmed=(reward is not None and self._last_reward is not None
                       and float(reward)>float(self._last_reward)+1e-5)
            # Reward semantics must be cross-checked in official code. If not
            # reliable, leave as UNKNOWN and never broadcast DISPOSED.
            key=self.latest_grasp_key
            if key in self.targets:
                self.targets[key].state='DISPOSED' if confirmed else 'SUSPECT_DROP'
            self.log(step,'DELIVERY_CONFIRMED' if confirmed else 'DROP_OR_UNKNOWN',key=key)
            if confirmed:self.disposed_seen+=1
            self.latest_grasp_key=None
        self._last_grasped=carrying
        if reward is not None:self._last_reward=float(reward)

    def advertised(self,step:int,localizer,phase,carrying):
        world=localizer.world()
        task_world=None
        if self.my_target in self.targets:
            w=localizer.world(self.targets[self.my_target].xy)
            if w is not None:task_world=[float(a) for a in w]
        entries=[]
        if localizer.anchored:
            for t in self.targets.values():
                if t.state not in ('KNOWN','CARRIED','DISPOSED'):continue
                w=localizer.world(t.xy)
                entries.append({'world':[float(a) for a in w],'state':t.state})
        c=localizer.world(self.center_local) if self.center_local is not None else None
        return {'schema':'v9.1','id':self.own_id,'step':int(step),
                'world_xy':None if world is None else [float(x) for x in world],
                'pose_anchored':bool(localizer.anchored and localizer.sigma_xy<25),
                'task_world':task_world,'carrying':bool(carrying),'phase':str(phase),
                'center_surface_world':None if c is None else [float(x) for x in c],
                'targets':entries[:24]}

class TaskScheduler:
    def __init__(self,own_id:int,hold_frames=110,allow_remote=None):
        self.me=int(own_id)
        self.hold_frames=int(hold_frames)
        self.allow_remote=(os.environ.get('V9_REMOTE_FETCH','1')=='1'
                           if allow_remote is None else bool(allow_remote))

    def decide(self,mem:TaskMemory,*,now:int,current_xy,carrying:bool,
               route_cost_fn=None,localizer=None,remaining_frames=5400,
               speed_empty=6.0,speed_loaded=2.8):
        if carrying:
            mem.my_target=None
            return None
        here=np.asarray(current_xy,dtype=float)
        options=[]
        for t in mem.targets.values():
            if t.state not in ('KNOWN','SUSPECT_DROP'):continue
            if not self.allow_remote and not t.local_observed:continue
            d=float(np.linalg.norm(t.xy-here))
            if route_cost_fn is not None:
                cost=route_cost_fn(t.xy)
                route_uncertain=cost is None or not math.isfinite(float(cost))
                # A relayed coordinate is a search lead, not an executable
                # fetch route. Keep exploring until the free-space component
                # reaches its vicinity, or a fresh local semantic ray sees it.
                if route_uncertain and not (t.local_observed and now-t.last_seen<=15):
                    continue
                cost=d*2.8 if route_uncertain else max(d,float(cost))
            else:
                route_uncertain=True;cost=d*2.8
            home=(float(np.linalg.norm(t.xy-mem.center_local)) if mem.center_local is not None else 600.)
            trip=(cost/max(.3,speed_empty)+30.+home/max(.3,speed_loaded)
                  +t.attempts*50.+(110. if route_uncertain else 0.))
            # Hard task completion feasibility deadline instead of arbitrary
            # history-based load penalties common to every candidate.
            over=max(0.,trip-remaining_frames)
            score=trip+2.5*over
            claim_block=False
            for peer in mem.peers.values():
                if (peer.target_world is None or not peer.raw.get('pose_anchored')):continue
                # No shared coordinates during unanchored segments.
                if not mem.targets or mem.my_target==t.key and peer.uid==self.me:continue
                # The adapter passes local candidate world through this helper.
                peer_local=localizer.local_from_world(peer.target_world) if localizer is not None else None
                if peer_local is None:continue
                if np.linalg.norm(peer_local-t.xy)<25 and (peer.carrying or peer.uid<self.me):
                    claim_block=True;break
            if not claim_block:options.append((score,t.key,t))
        if not options:
            mem.my_target=None
            return None
        options.sort(key=lambda z:(z[0],z[1]))
        inc=next((a for a in options if a[1]==mem.my_target),None)
        best=options[0]
        if inc is not None and (now-mem._last_claim_change<self.hold_frames or
                                inc[0]-best[0] < max(70.,.18*inc[0])):
            return inc[2]
        if mem.my_target!=best[1]:
            mem.log(now,'ASSIGN',key=best[1],eta=round(best[0],1))
            mem.my_target=best[1];mem._last_claim_change=int(now)
        return best[2]
