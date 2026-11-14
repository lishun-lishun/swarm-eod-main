"""Incremental V2 fusion. Exactly one selected V2 task/planner/action pipeline."""
import math
import os
import numpy as np
from .pose_sict_drone import SictRescueDrone, CELL, V_CARRY, wrap
from .pose_fusion_switches import enabled
from .pose_fusion_delivery import ApproachSet

class FusionRescueDrone(SictRescueDrone):
    # ------------------------------------------------------------ module E
    # No-GPS heading correction.  The odometer rotation increment has about
    # 1 deg noise per frame, so dead-reckoned heading random-walks and rotates
    # every later displacement (measured: carriers in no-GPS zones had a median
    # 78 px pose error, 6 px with GPS).  While the compass works, the drone
    # learns the dominant wall direction modulo 90 deg from its own LiDAR; in
    # no-GPS zones that learned axis bounds the heading error.  Public sensors
    # only; no map prior is hard-coded.
    _E_GAIN = 0.15
    _E_MAX_ERR = math.radians(20.0)

    def _wall_axis(self):
        """Dominant wall direction in the body frame, modulo 90 deg."""
        lidar = self.lidar_values()
        if lidar is None or self.lidar_is_disabled():
            return None
        d = np.asarray(lidar, dtype=float)
        a = np.asarray(self.lidar_rays_angles(), dtype=float)
        n = min(len(d), len(a))
        d, a = d[:n], a[:n]
        x, y = d * np.cos(a), d * np.sin(a)
        k = 3
        ok = (d[:-k] < 190.0) & (d[k:] < 190.0)
        dx, dy = x[k:] - x[:-k], y[k:] - y[:-k]
        seg = np.hypot(dx, dy)
        # Same straight wall: short, consistent span between the two rays.
        ok &= (seg > 6.0) & (seg < 45.0)
        if np.count_nonzero(ok) < 12:
            return None
        w = seg[ok]
        z = np.sum(w * np.exp(4j * np.arctan2(dy[ok], dx[ok])))
        total = float(np.sum(w))
        if abs(z) < 0.55 * total:  # walls in view disagree: no correction
            return None
        return float(np.angle(z)) / 4.0, abs(z)

    def _update_pose(self):
        super()._update_pose()
        if not enabled('E') or self.pose is None:
            return
        est = self._wall_axis()
        if est is None:
            return
        psi_b, weight = est
        if not hasattr(self, '_e_axis'):
            self._e_axis = 0j
        if self.measured_compass_angle() is not None:
            self._e_axis = 0.995 * self._e_axis + weight * complex(
                math.cos(4.0 * (self.heading + psi_b)), math.sin(4.0 * (self.heading + psi_b)))
            return
        if abs(self._e_axis) < 2000.0:  # axis not learned yet
            return
        theta0 = math.atan2(self._e_axis.imag, self._e_axis.real) / 4.0
        q = math.pi / 2.0
        err = (self.heading + psi_b - theta0 + q / 2.0) % q - q / 2.0
        if abs(err) < self._E_MAX_ERR:
            self.heading = wrap(self.heading - self._E_GAIN * err)

    # ------------------------------------------------------------ module C
    # Cargo-aware path following.  Measured on E runs: carriers move at a
    # median 1.7-2.0 px/frame against a 3.25 px/frame loaded steady speed,
    # while the route detour is only 1.1-1.4x.  C1 looks further along the
    # path (straight chords still keep the 32 px bomb clearance); C2 lowers
    # the loaded wall-repulsion margin that pushes carriers off the line.
    _C_REACH = float(os.environ.get('FUSION_C_REACH', '90'))
    _C_MARGIN = float(os.environ.get('FUSION_C_MARGIN', '16'))

    def _waypoint(self, min_clear=14.0, reach=110.0):
        if enabled('C1') and getattr(self, 'was_carrying', False) and reach < self._C_REACH:
            reach = self._C_REACH
        return super()._waypoint(min_clear=min_clear, reach=reach)

    def _velocity_to_command(self, v_des, lidar, angles, carrying, careful=False,
                             wall_margin=None, ignore_target=None):
        if (enabled('C2') and carrying and not careful and wall_margin is not None
                and wall_margin > self._C_MARGIN):
            wall_margin = self._C_MARGIN
        return super()._velocity_to_command(v_des, lidar, angles, carrying, careful=careful,
                                            wall_margin=wall_margin, ignore_target=ignore_target)

    # ------------------------------------------------------------ module B
    def _setup(self):
        super()._setup()
        self.approaches=ApproachSet()
        self._approach_version=-1
        self._dock_since=None
        self._dock_style=0
        self._verify_until=-1
        self._last_held_relative=None
        self._last_command={'forward':0.,'lateral':0.,'rotation':0.,'grasper':0}
        self._cargo_mode='pull'
        self._cargo_mode_since=-1000
        self._cargo_lock_until=-1
        self._cargo_locked=False
        self._pose_pickup_key=None
        self._pose_stuck_seen=0
        self._pose_stuck_base=0

    def _direction_clearance(self,lidar,angles,world_angle):
        """Robust free distance around one world direction from legal LiDAR."""
        if lidar is None or angles is None or len(lidar)==0:
            return 230.0
        rel=np.abs((angles-(world_angle-self.heading)+math.pi)%(2*math.pi)-math.pi)
        cone=lidar[rel<.34]
        if cone.size==0:
            return 230.0
        # One noisy short ray must not flip the whole cargo train.
        return float(np.percentile(cone,25))

    def _cargo_pose(self,step,move_angle,lidar,angles,force_push=False):
        """Choose push/pull and return a bounded yaw command plus alignment.

        Pulling is stable on an ordinary route.  Near an obstacle, choose the
        side with room for the bomb.  Hysteresis prevents repeated 180-degree
        sweeps in a narrow passage.  Docking always overrides the choice and
        puts the bomb in front of the drone.
        """
        front=self._direction_clearance(lidar,angles,move_angle)
        rear=self._direction_clearance(lidar,angles,move_angle+math.pi)
        wanted='push' if force_push else self._cargo_mode
        if wanted!=self._cargo_mode:
            self._cargo_mode=wanted; self._cargo_mode_since=step
            self._event('cargo_pose',step,mode=wanted,
                        front=round(front),rear=round(rear))
        cargo_angle=move_angle if self._cargo_mode=='push' else move_angle+math.pi
        desired=wrap(cargo_angle-(self.cargo_bearing or 0.0))
        err=wrap(desired-self.heading)
        return self._yaw_for_cargo(cargo_angle),abs(err)<.38

    def _corridor_pose(self,step,move_angle):
        """Select one longitudinal pose before a squeeze, then lock it."""
        here=self.grid.clearance_at(*self.pose)
        upcoming=[]
        if self.path:
            lo=max(0,getattr(self,'path_i',0))
            for p in self.path[lo:min(len(self.path),lo+14)]:
                if math.dist(self.pose,p)<=150.:
                    upcoming.append(self.grid.clearance_at(*p))
        ahead=min(upcoming,default=100.)
        if self._cargo_locked:
            if step>=self._cargo_lock_until and here>58. and ahead>52.:
                self._cargo_locked=False
            return self._cargo_locked
        # Make the decision while the complete drone-bomb train still has
        # turning room, never after it is already wedged in the slot.
        if here>58. and ahead<40.:
            cargo_now=wrap(self.heading+(self.cargo_bearing or 0.))
            push_err=abs(wrap(move_angle-cargo_now))
            pull_err=abs(wrap(move_angle+math.pi-cargo_now))
            self._cargo_mode='push' if push_err+.12<pull_err else 'pull'
            self._cargo_mode_since=step
            self._cargo_lock_until=step+260
            self._cargo_locked=True
            self._event('cargo_corridor_lock',step,mode=self._cargo_mode,
                        here=round(here),ahead=round(ahead))
        return self._cargo_locked

    def _level_yaw(self):
        """Gently settle the hull on the nearest horizontal heading."""
        target=0.0 if math.cos(self.heading)>=0.0 else math.pi
        err=wrap(target-self.heading)
        return max(-.28,min(.28,err/.35*.28))

    def _pose_carry_command(self,step,cmd,lidar,angles):
        """Apply the pose layer to the C2/base carrier action generator."""
        if (step<self.backoff_until or step<self.escape_until
                or self.cargo_bearing is None):
            return cmd
        f=float(cmd.get('forward',0.)); l=float(cmd.get('lateral',0.))
        ca,sa=math.cos(self.heading),math.sin(self.heading)
        wx,wy=ca*f-sa*l,sa*f+ca*l
        if math.hypot(wx,wy)<.08:
            return cmd
        move=math.atan2(wy,wx)
        pickup=getattr(self,'pickup_key',None)
        if pickup!=self._pose_pickup_key:
            self._pose_pickup_key=pickup
            self._cargo_mode='pull'; self._cargo_locked=False
            self._cargo_lock_until=-1; self._cargo_mode_since=step
            self._pose_stuck_seen=self.stuck_count
            self._pose_stuck_base=self.stuck_count
        near=min((h[2] for h in self.center_hits),default=math.inf)
        force_push=near<120.
        if not force_push and self.path:
            i=min(len(self.path)-1,max(0,getattr(self,'path_i',0))+3)
            target=self.path[i]
            if math.dist(self.pose,target)>2.:
                move=math.atan2(target[1]-self.pose[1],target[0]-self.pose[0])
        # Preserve the proven terminal sweep only as a bounded fallback after
        # the direct push has had time to work.
        if (force_push and near<40. and self.push_since is not None
                and step-self.push_since>45):
            return cmd
        # The proven base controller already pushes on the live centre hit.
        # Do not replace that terminal action with an angle inferred at pickup.
        if force_push:
            return cmd
        # Pose adaptation is reserved for the semifinal's degraded-sensor
        # regions. On ordinary maps the verified C2 action is byte-for-byte
        # preserved, avoiding regressions caused by unnecessary 180° turns.
        degraded=(self.measured_gps_position() is None
                  or self.communicator_is_disabled())
        if not degraded:
            self._cargo_mode='pull'
            return cmd
        # A short LiDAR return is exactly the wrong place to rotate the bomb:
        # the 180° sweep hooks it on the nearby wall.  Request an alternative
        # pose only after three failed recoveries, then execute it at the first
        # genuinely open staging point.
        lidar_open=(lidar is None or float(np.min(lidar))>72.)
        map_open=self.grid.clearance_at(*self.pose)>58.
        repeated=self.stuck_count-self._pose_stuck_base>=3
        if (repeated and self.stuck_count>self._pose_stuck_seen
                and step>=self.escape_until and lidar_open and map_open):
            self._pose_stuck_seen=self.stuck_count
            self._cargo_mode='push' if self._cargo_mode=='pull' else 'pull'
            self._cargo_mode_since=step
            self._cargo_lock_until=step+240
            self._event('cargo_degraded_recovery',step,mode=self._cargo_mode,
                        clearance=round(self.grid.clearance_at(*self.pose)))
        if step>=self._cargo_lock_until:
            return cmd
        rot,aligned=self._cargo_pose(step,move,lidar,angles,force_push=False)
        cmd['rotation']=rot if force_push else .65*rot
        if not aligned and abs(rot)>.8:
            cmd['forward']*=.55; cmd['lateral']*=.55
        return cmd

    def control(self):
        cmd=super().control()
        if self.ready:
            # In open space, keep unloaded cruise visually and dynamically
            # level. Fetch alignment and all cargo pose commands take priority.
            if (enabled('H') and os.environ.get('FUSION_LEVEL','0')=='1'
                    and not self.grasped_bombs()
                    and self.state in ('EXPLORE','RETURN')
                    and abs(cmd.get('rotation',0.))<.05):
                lidar,_=self._lidar()
                f=float(cmd.get('forward',0.)); l=float(cmd.get('lateral',0.))
                ca,sa=math.cos(self.heading),math.sin(self.heading)
                wx,wy=ca*f-sa*l,sa*f+ca*l
                # Level only while the requested motion is already close to
                # horizontal, so posture control never fights a vertical path.
                if ((lidar is None or float(np.min(lidar))>90.)
                        and math.hypot(wx,wy)>.15
                        and abs(math.sin(math.atan2(wy,wx)))<.42):
                    cmd['rotation']=self._level_yaw()
            self._last_command=dict(cmd)
        return cmd

    def define_message_for_all(self):
        msg=super().define_message_for_all()
        if msg is not None and enabled('B'):
            msg['approaches']=self.approaches.payload()
        return msg

    def _perceive(self,step):
        dynamic=super()._perceive(step)
        if enabled('B'):
            self.approaches.observe(self.grid,self.pose,self.center_hits,step)
            if not self.communicator_is_disabled():
                for _,msg in self.communicator.received_messages or ():
                    if isinstance(msg,dict) and msg.get('v3')==1:
                        self.approaches.merge(self.grid,msg.get('approaches',()),step)
        return dynamic

    def _refresh_center_field(self,step):
        if not enabled('B') or not self.was_carrying: return super()._refresh_center_field(step)
        if self.center_cum is not None and step-self.center_step<25 and self._approach_version==self.approaches.version: return
        good=self.approaches.valid(self.grid,step)
        if not good:
            self.center_cum=self.center_mcp=None; self.center_step=step
            return
        # Multi-source shortest path chooses by actual V2 route cost. Prefer
        # close, roomy witnessed positions to distant center observations.
        good.sort(key=lambda w:math.dist(w[:2],w[2:4])-.3*min(50.,self.grid.clearance_at(*w[:2])))
        goals=sorted({self.grid.cell(*w[:2]) for w in good[:32]})
        self.center_mcp,self.center_cum=self.grid.dijkstra('carry',goals)
        self.center_step=step; self._approach_version=self.approaches.version

    def _transitions(self,step,carrying):
        if not enabled('B'): return super()._transitions(step,carrying)
        if self.was_carrying and not carrying:
            self.state='VERIFY'
            # Public one-frame reward is set by the official disposal callback;
            # held Bomb objects are never inspected. Proximity is not proof.
            confirmed=float(self.reward)>0
            if confirmed:
                self.know.disposed+=1
                self._event('delivery_confirmed',step)
            elif self.yielded_cargo:
                self.know.drops.add(getattr(self,'pickup_key',''))
                self._event('delivery_handover',step)
            else:
                self.know.drops.add(getattr(self,'pickup_key',''))
                self.stats['drops']+=1
                near=[p for p,d in self.visible_bombs if d<65]
                if near:
                    p=min(near,key=lambda p:math.dist(p,self.pose))
                    self.know.observe_bomb(p,step+1,1.,view=(*self.pose,math.dist(p,self.pose)))
                    self._event('delivery_drop_visible',step)
                else:
                    self._event('delivery_release_unresolved',step)
                    self._verify_until=step+12
            self.was_carrying=False; self.cargo_bearing=None; self.yielded_cargo=False
            self.path=[]; self.cum_step=-1000; self.history.clear(); self._reset_progress(); self._dock_since=None
            return
        super()._transitions(step,carrying)

    def _verify(self,step,cmd,lidar,angles):
        self.state='VERIFY'
        # Brief sensor scan for a released bomb. No speculative disposal count.
        cmd['grasper']=0; cmd['rotation']=.2
        cmd['forward'],cmd['lateral']=self._velocity_to_command((0.,0.),lidar,angles,False)
        if any(d<65 for _,d in self.visible_bombs): self._verify_until=step
        return cmd

    def _fetch(self,step,cmd,lidar,angles):
        if enabled('B') and step<self._verify_until: return self._verify(step,cmd,lidar,angles)
        return super()._fetch(step,cmd,lidar,angles)

    def _explore(self,step,cmd,lidar,angles):
        if enabled('B') and step<self._verify_until: return self._verify(step,cmd,lidar,angles)
        return super()._explore(step,cmd,lidar,angles)

    def _carry(self,step,cmd,lidar,angles):
        if not enabled('B'):
            cmd=super()._carry(step,cmd,lidar,angles)
            return self._pose_carry_command(step,cmd,lidar,angles) if enabled('H') else cmd
        cmd['grasper']=1
        # Keep V2's observed co-carry ownership protection.
        mine=self.pickup_info
        for pid,p in self.know.peers.items():
            other=p.get('pickup')
            if pid<self.me and p.get('carry') and other and mine and math.dist(other[:2],mine[:2])<25 and abs(other[2]-mine[2])<80:
                self.yielded_cargo=True; self.no_grasp_until=step+60
                cmd['grasper']=0; self._event('cocarry_release',step,partner=pid)
                return cmd
        if step<self.backoff_until and self.backoff_dir is not None:
            self.state='RECOVER'
            cmd['forward'],cmd['lateral']=self._velocity_to_command(tuple(2.5*x for x in self.backoff_dir),lidar,angles,True)
            return cmd
        live=[h for h in self.center_hits if h[2]<110 and self._center_passable(h[:2])]
        if live and not self.approaches.blocked(self.pose,step):
            self.state='FINAL_DELIVERY'
            near=min(live,key=lambda h:h[2]); a=math.atan2(near[1]-self.pose[1],near[0]-self.pose[0])
            if self._dock_since is None: self._dock_since=step
            elapsed=step-self._dock_since
            # Straight/pivot alignment won the physics sweep's success test.
            # After no progress, bounded backoff and another witness precede retry.
            v=(2.5*math.cos(a),2.5*math.sin(a))
            cmd['rotation'],aligned=self._cargo_pose(step,a,lidar,angles,force_push=True)
            if not aligned:
                # Rotate the bomb to the leading side before translating; this
                # avoids dragging it past the disposal face during the turn.
                v=(.45*math.cos(a),.45*math.sin(a))
            if near[2]<38 and elapsed>45:
                cmd['rotation']=.65 if self._dock_style%2==0 else -.65
                v=(1.5*math.cos(a)-.5*math.sin(a),1.5*math.sin(a)+.5*math.cos(a))
            ignore=None
            if lidar is not None:
                rel=np.abs((angles-(a-self.heading)+math.pi)%(2*math.pi)-math.pi)
                ignore=(rel<.45)&(lidar>near[2]-12)
            cmd['forward'],cmd['lateral']=self._velocity_to_command(v,lidar,angles,True,careful=True,wall_margin=14.,ignore_target=ignore)
            if elapsed>130:
                self.approaches.fail(self.pose,step); self._dock_style+=1; self.stats['push_retries']+=1
                self._event('approach_failed',step,style=self._dock_style)
                self._dock_since=None; self.center_step=-1000; self.center_cum=None; self.path=[]
                self.backoff_dir=(-math.cos(a),-math.sin(a)); self.backoff_until=step+30
                self.state='RECOVER'
            return cmd
        self._dock_since=None
        self.state='APPROACH_CENTER'
        if not self.path or self.path_goal!='center' or step-self.path_step>25:
            path=self._center_path(step)
            if path:
                self.path=path; self.path_i=0; self.path_goal='center'; self.path_step=step
            else:
                # Seek a new observation with the existing V2 exploration
                # selector and planner; no second planner/state machine.
                self.state='CARRY'
                if self.carry_goal is None or math.dist(self.pose,self.carry_goal)<30 or step-self.carry_goal_step>200:
                    self._refresh_dijkstra(step,force=True); choice=self._explore_target(step)
                    self.carry_goal=choice[0] if choice else None; self.carry_goal_step=step
                    if choice: self._set_path_to(choice[0],step,end_cell=choice[1])
                elif not self.path or step-self.path_step>30:
                    self._refresh_dijkstra(step); self._set_path_to(self.carry_goal,step)
                self.path_goal='carry_explore'
        wp=self._waypoint(min_clear=32.,reach=42.)
        field=None
        if self.center_cum is not None:
            c=self.grid.cell(*self.pose); field=float(self.center_cum[c])*CELL
        goal=self.path[-1] if self.path else self.carry_goal
        if self._progress_stalled(step,field,window=160) or self._stuck_monitor(step,goal):
            self._start_escape(step,lidar,angles,away_from=wp); self.state='RECOVER'; return cmd
        v=(0.,0.) if wp is None else self._toward(wp,V_CARRY)
        cmd['forward'],cmd['lateral']=self._velocity_to_command(v,lidar,angles,True,wall_margin=22.)
        if wp is not None and math.hypot(*v)>.1:
            move=math.atan2(v[1],v[0])
            pre_dock=any(h[2]<120 for h in self.center_hits)
            rot,aligned=self._cargo_pose(step,move,lidar,angles,force_push=pre_dock)
            cmd['rotation']=.65*rot
            if not aligned:
                # A 180-degree pose change sweeps the bomb around the hull.
                # Slow down until it is longitudinal again.
                cmd['forward']*=.28; cmd['lateral']*=.28
        return cmd


