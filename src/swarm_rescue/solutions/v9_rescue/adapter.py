"""Official DroneAbstract adapter. This is a COMPLETE alternative controller.

IMPORTANT: Import only inside the competition project. Unit tests target the
pure-Python core separately, without importing Arcade/Pymunk. Test in official
Launcher before selecting V9 in the submission entrypoint.
"""
from __future__ import annotations
import math
import numpy as np
from collections import deque
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from .config import V9Config
from .geometry import Pose, RelativeDetection, rotate, wrap, norm, normalize
from .localization import EgoLocalizer, correlative_scan_match
from .mapping import BeliefGrid
from .tasks import TaskMemory, TaskScheduler
from .planning import KnownFreePlanner, RouteFollower, EgoBreadcrumbs, lidar_unknown_step
from .mission import MissionSupervisor
from .control import ModelAwareMotion, make_reference
from .cargo import CargoSwingTracker
from .diagnostics import register as register_diagnostics

KIND=DroneSemanticSensor.TypeEntity

class V9RescueDrone(DroneAbstract):
    """One owner of task, route and output. No V4-V8 mixins or action overrides."""
    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier,misc_data=misc_data,**kwargs)
        self.cfg=V9Config()
        self.deadline_frames=(int(misc_data.max_timestep_limit)
                              if misc_data is not None and
                              misc_data.max_timestep_limit is not None
                              else self.cfg.max_deadline_frames)
        self.loc=EgoLocalizer()
        size=self.size_area or (2000.,2000.)
        # Double local map extent: local frame begins at a robot, not arena center.
        self.grid=BeliefGrid((size[0]*2.,size[1]*2.),self.cfg)
        self.mem=TaskMemory(int(self.identifier or 0),self.cfg.track_match_px)
        self.assigner=TaskScheduler(int(self.identifier or 0))
        self.planner=KnownFreePlanner(self.grid)
        self.follow=RouteFollower()
        self.breadcrumbs=EgoBreadcrumbs()
        self.mission=MissionSupervisor()
        self.motion=ModelAwareMotion(self.cfg)
        self.cargo_swing=CargoSwingTracker()
        self._cached_lidar=None
        self._cached_angles=None
        self._last_phase='BOOT'
        self._active_goal=None
        self._active_frontier=None
        self._active_frontier_since=-1
        self._frontier_reached_since=-1
        self._last_plan=-999
        self._last_local_map_revision=-1
        self._last_route_target=None
        self._last_frontier_plan=-100000
        self._scan_phase=0
        self._last_move=deque(maxlen=95)
        self._end_game=False
        self._latest_events=[]
        self._raw_odom=None
        self._trace=deque(maxlen=2400)
        register_diagnostics(self)

    def define_message_for_all(self):
        if self.communicator_is_disabled():return None
        return self.mem.advertised(int(self.elapsed_timestep),self.loc,
                                   self.mission.phase,bool(self.grasped_bombs()))

    @staticmethod
    def _clean_lidar(raw,raw_angles):
        if raw is None or raw_angles is None:return None,None
        try:
            rr=np.asarray(raw,dtype=float).ravel()
            aa=np.asarray(raw_angles,dtype=float).ravel()
            if rr.shape!=aa.shape or rr.size<40:return None,None
            return rr,aa
        except (ValueError,TypeError):return None,None

    @staticmethod
    def _sensor_kind(obj):
        if obj.entity_type==KIND.BOMB:return 'bomb'
        if obj.entity_type==KIND.DRONE:return 'drone'
        if obj.entity_type==KIND.DISPOSAL_CENTER:return 'center'
        return None

    def _read_detections(self):
        result=[]
        for obj in self.semantic_values() or ():
            try:
                d,a=float(obj.distance),float(obj.angle)
                typ=self._sensor_kind(obj)
                if typ is not None and math.isfinite(d) and math.isfinite(a) and 0<d<=self.cfg.semantic_range_px+5:
                    result.append(RelativeDetection(typ,d,a,bool(obj.grasped)))
            except (TypeError,ValueError,AttributeError):continue
        return result

    def _cluster_semantic(self,det,pose):
        """A bomb may hit 2-3 rays. Return one local candidate per object."""
        bomb=[];centers=[];peers=[]
        for h in det:
            if h.kind=='drone':peers.append((float(h.distance),float(h.angle),False));continue
            if h.kind=='center':centers.append(h);continue
            if h.kind!='bomb' or h.grasped:continue
            direction=pose.yaw+h.angle
            center=np.array((pose.x+(h.distance+11.)*math.cos(direction),
                             pose.y+(h.distance+11.)*math.sin(direction)))
            if not any(np.linalg.norm(center-p)<21 for p,_ in bomb):
                bomb.append((center,h))
        return bomb,centers,peers

    def _local_link(self,a,b,rr,aa,carrying=False):
        """Local short segment: live lidar AND mapped known-free corridor.

        Cannot certify longer routes. Never crosses unknown or map obstacle.
        """
        if not self.grid.known_free_segment(a,b,min_clear_px=(23 if carrying else 17)):
            return False
        d=math.dist(a,b)
        if d<1.0:return True
        if d>65.0:return False
        direction=math.atan2(b[1]-a[1],b[0]-a[0])-self.loc.pose.yaw
        angular=np.abs((aa-direction+math.pi)%(2*math.pi)-math.pi)
        selected=rr[(angular<.13)&np.isfinite(rr)]
        if selected.size<2:return False
        radius=self.cfg.loaded_radius_px if carrying else self.cfg.empty_radius_px
        return float(np.quantile(selected,.2))>d+radius+self.cfg.current_collision_margin_px

    def _record_perception(self,pose,det,step):
        bombs,centers,peers=self._cluster_semantic(det,pose)
        for center,h in bombs:self.mem.observe_bomb(center,step,view_xy=pose.xy())
        if centers:
            h=min(centers,key=lambda x:x.distance)
            direction=pose.yaw+h.angle
            p=(pose.x+h.distance*math.cos(direction),pose.y+h.distance*math.sin(direction))
            self.mem.observe_center(p,step)
        return bombs,centers,peers

    def _local_explore_direction(self,pose,rr,aa,preferred_xy=None):
        """Deterministic sensor-driven explore fallback; not a hidden-map waypoint."""
        direction=-math.pi+2*math.pi*((int(self.identifier or 0)*0.6180339887)%1)
        sectors=[]
        recent_breadcrumbs=[xy for _,xy in list(self.breadcrumbs.points)[-80::8]]
        for k in range(16):
            theta=wrap(direction+2*math.pi*k/16)
            diff=np.abs((aa-theta+math.pi)%(2*math.pi)-math.pi)
            r=rr[(diff<.18)&np.isfinite(rr)]
            clear=float(np.quantile(r,.20)) if r.size else 0.
            # Bias against previously visited directions through short-term trace.
            step=pose.xy()+max(0.,clear-25.)*rotate((1.,0.),pose.yaw+theta)
            repeats=sum(max(0.,1.-math.dist(step,hp)/125.) for hp in recent_breadcrumbs)
            goal_bias=0.
            if preferred_xy is not None:
                bearing=wrap(math.atan2(preferred_xy[1]-pose.y,
                                         preferred_xy[0]-pose.x)-pose.yaw)
                goal_bias=55.*math.cos(wrap(theta-bearing))
            sectors.append((clear-6.*repeats+goal_bias,theta,clear))
        _,th,clear=max(sectors)
        return th,clear

    def _plan_goal(self,*,step,pose,mission_goal,carrying,rr,aa):
        if mission_goal is None:
            self._active_goal=None
            return None
        current=tuple(float(v) for v in pose.xy())
        target=tuple(float(v) for v in mission_goal)
        if (self._last_route_target is not None and
            math.dist(target,self._last_route_target)<18 and
            not self.follow.stalled(step) and self.follow.route.points):
            return self.follow.waypoint(current,now=step,
                local_segment_ok=lambda a,b:self._local_link(a,b,rr,aa,carrying))
        if step-self._last_plan<self.cfg.plan_every and self.follow.route.points:
            return self.follow.waypoint(current,now=step,
                local_segment_ok=lambda a,b:self._local_link(a,b,rr,aa,carrying))
        self._last_plan=step
        path=self.planner.path(current,target,carrying=carrying,
              local_link=lambda a,b:self._local_link(a,b,rr,aa,carrying))
        if path.status=='CONNECTED' and self.follow.set_route(path,step):
            self._last_route_target=target
            return self.follow.waypoint(current,now=step,
                local_segment_ok=lambda a,b:self._local_link(a,b,rr,aa,carrying))
        self._last_route_target=target
        self.follow.reset()
        return None

    def control(self):
        now=int(self.elapsed_timestep)
        cmd={'forward':0.,'lateral':0.,'rotation':0.,
             'grasper':int(bool(self.grasped_bombs()))}
        try:
            return self._step(now)
        except (ValueError,TypeError,ArithmeticError) as exc:
            # Fail safely and log; never swallow bugs in development testing.
            self._latest_events.append({'step':now,'kind':'V9_EXCEPTION','detail':repr(exc)})
            if len(self._latest_events)>100:self._latest_events.pop(0)
            return cmd

    def _step(self,step):
        rr,aa=self._clean_lidar(self.lidar_values(),self.lidar_rays_angles())
        if rr is None:
            return {'forward':0.,'lateral':0.,'rotation':.22,'grasper':int(bool(self.grasped_bombs()))}
        self._cached_lidar=rr;self._cached_angles=aa
        odom=self.odometer_values()
        self._raw_odom=odom
        gps=self.measured_gps_position()
        comp=self.measured_compass_angle()
        st=self.loc.predict(step=step,odom=odom,gps=gps,compass=comp)
        pose=st.pose
        # Match against OLD map before integrating this frame's own scan.
        if (step>15 and step%self.cfg.scan_match_period==0 and
            (gps is None or st.sigma_xy>9.0)):
            correction=correlative_scan_match(self.grid,pose,rr,aa,
                            max_range=self.cfg.lidar_range_px,
                            min_hits=self.cfg.scan_match_min_hits)
            if correction is not None:
                self.loc.accept_scan_match(dx=correction[0],dy=correction[1],
                    dyaw=correction[2],improvement=correction[3],
                    static_correspondences=correction[4])
                pose=self.loc.pose
        det=self._read_detections()
        bomb_obs,centers,near_peers=self._record_perception(pose,det,step)
        # Radio messages are from actual communication radius only.
        messages=() if self.communicator_is_disabled() else self.communicator.received_messages
        self.mem.read_messages(messages,step,self.loc)
        carry=bool(self.grasped_bombs())
        reward=float(self.reward) if getattr(self,'reward',None) is not None else None
        self.mem.observe_grasp_transition(step,carry,reward=reward,at_xy=pose.xy())
        cargo_bearing,cargo_uncertainty=self.cargo_swing.observe(
            carrying=carry,body_yaw=pose.yaw,
            pickup_bearing=self.mission.cargo_bearing)
        self.breadcrumbs.add(pose.xy(),step)
        if step%self.cfg.map_every==0:
            self.grid.integrate_lidar(pose,rr,aa,step,dynamic_detections=det)
            self.grid.integrate_semantic_geometry(pose,rr,aa,step)
        self._last_move.append((step,pose.xy()))
        if len(self._last_move)==95 and norm(pose.xy()-self._last_move[0][1])<8:
            self._latest_events.append({'step':step,'kind':'LOW_PROGRESS','phase':self.mission.phase})
        # One geodesic distance field for all legal known target bids.
        route_cost_fn=None
        schedule_event=(step%self.cfg.plan_every==0 or
                        any(event['step']==step and event['kind'] in
                            ('FIRST_SEEN','GRASP_CONFIRMED','DELIVERY_CONFIRMED',
                             'DROP_OR_UNKNOWN','PEER_CARRIED','PEER_DISPOSED')
                            for event in self.mem.events[-8:]))
        if not carry and self.mem.targets and schedule_event:
            field,_=self.planner.distances_from(pose.xy(),
                local_link=lambda a,b:self._local_link(a,b,rr,aa,False))
            def route_cost_fn(xy):
                c=self.grid.cell(xy)
                if c is None:return math.inf
                # Bomb surface cells can be occupied or still unknown. A
                # connected observation position within sensor range suffices.
                return min((cost+math.dist(self.grid.xy(cell),xy)
                            for cell,cost in field.items()
                            if abs(cell[0]-c[0])<=5 and abs(cell[1]-c[1])<=5),
                           default=math.inf)
        # Only plan expensive target distances when there is a known target and
        # a scheduling event; do NOT run multiple grid searches every frame.
        event=schedule_event
        if not carry and event and len(self.mem.targets):
            selected=self.assigner.decide(self.mem,now=step,current_xy=pose.xy(),
                  carrying=False,route_cost_fn=route_cost_fn,localizer=self.loc,
                  remaining_frames=max(0,self.deadline_frames-step),
                  speed_empty=self.cfg.empty_speed_px_frame,
                  speed_loaded=self.cfg.loaded_speed_px_frame)
        else:
            selected=self.mem.targets.get(self.mem.my_target)
        # Match live detection only to selected task; other bombs remain known
        # but cannot arbitrarily preempt a physically committed fetch.
        live=[]
        if selected is not None:
            for position,h in bomb_obs:
                if norm(position-selected.xy)<28.0:live.append(h)
        # If task selection is blocked by an inconsistent report, continuing
        # exploration remains possible rather than blindly grabbing it.
        frontier_target=None
        if (carry and self.mem.center_local is None) or (not carry and selected is None):
            frontier_reached=(self._active_frontier is not None
                    and math.dist(pose.xy(),self._active_frontier.world)<17.)
            frontier_timed_out=(self._active_frontier is not None and
                    step-self._active_frontier_since>240)
            if frontier_reached and self._frontier_reached_since<0:
                self._frontier_reached_since=step
            if not frontier_reached:
                self._frontier_reached_since=-1
            if self._frontier_reached_since>=0 and step-self._frontier_reached_since>55:
                frontier_timed_out=True
            if frontier_timed_out and self._active_frontier is not None:
                self.planner.on_frontier_finished(self._active_frontier,step,
                                                   gained_cells=0,arrived=False)
                self._active_frontier=None
                self._frontier_reached_since=-1
            if step-self._last_frontier_plan>=self.cfg.plan_every and (
                    self._active_frontier is None):
                self._last_frontier_plan=step
                if self._active_frontier is not None:
                    reached=math.dist(pose.xy(),self._active_frontier.world)<17.
                    self.planner.on_frontier_finished(self._active_frontier,step,
                                                        gained_cells=0,arrived=reached)
                goal=self.planner.select_frontier(tuple(pose.xy()),now=step,
                    drone_id=int(self.identifier or 0),
                    local_link=lambda a,b:self._local_link(a,b,rr,aa,carry),
                    peer_destinations=())
                self._active_frontier=goal
                self._active_frontier_since=step
                self._frontier_reached_since=-1
            frontier_target=self._active_frontier.world if self._active_frontier is not None else None
        intent=self.mission.step(now=step,pose=pose,carrying=carry,
                                 visible_bombs=live,visible_centers=centers,
                                 task_xy=None if selected is None else selected.xy,
                                 remembered_center_xy=self.mem.center_local,
                                 explore_goal_xy=frontier_target)
        if intent.phase!=self._last_phase:
            self.follow.reset();self._last_route_target=None
            self._last_phase=intent.phase
        direct_live=(intent.phase=='FETCH' and intent.terminal_target is not None)
        direct_center=(intent.phase=='DOCK')
        waypoint=None
        # A frontier reached is not an exploration completion: advance a few
        # pixels toward UNKNOWN under current LiDAR, then remap/replan.
        near_frontier=(self._active_frontier is not None and
                       math.dist(pose.xy(),self._active_frontier.world)<18.)
        if near_frontier and intent.phase=='EXPLORE':
            angle=wrap(self._active_frontier.into_unknown_yaw-pose.yaw)
            waypoint=lidar_unknown_step(pose,rr,aa,
                body_radius_px=self.cfg.empty_radius_px,
                yaw_relative=angle,max_step_px=self.cfg.max_unknown_step_px)
        if waypoint is None and intent.goal_xy is not None and (direct_live or direct_center):
            # Direct relative sightings still need the final local safety layer.
            waypoint=intent.goal_xy
        elif waypoint is None and intent.goal_xy is not None:
            waypoint=self._plan_goal(step=step,pose=pose,
                 mission_goal=intent.goal_xy,carrying=carry,rr=rr,aa=aa)
        if waypoint is None and intent.goal_xy is not None:
            # GPS failure does not erase ego-local breadcrumbs.
            if carry:
                yaw=wrap(math.atan2(intent.goal_xy[1]-pose.y,
                                    intent.goal_xy[0]-pose.x)-pose.yaw)
                waypoint=lidar_unknown_step(pose,rr,aa,
                    body_radius_px=self.cfg.loaded_radius_px,
                    yaw_relative=yaw,max_step_px=18.)
                if waypoint is None:
                    waypoint=self.breadcrumbs.local_backtrack(pose.xy(),
                        current_safe=lambda a,b:self._local_link(a,b,rr,aa,True))
                    if waypoint is not None and math.dist(pose.xy(),waypoint)<10.:
                        waypoint=None
            elif self._active_frontier is not None:
                # At a verified frontier, probe into its known unknown-facing side.
                if math.dist(pose.xy(),self._active_frontier.world)<18.:
                    yaw=wrap(self._active_frontier.into_unknown_yaw-pose.yaw)
                    waypoint=lidar_unknown_step(pose,rr,aa,
                        body_radius_px=self.cfg.empty_radius_px,
                        yaw_relative=yaw,max_step_px=self.cfg.max_unknown_step_px)
        if (intent.goal_xy is not None and intent.phase in ('FETCH','CARRY')
                and (waypoint is None or math.dist(pose.xy(),waypoint)<8.)
                and math.dist(pose.xy(),intent.goal_xy)>15.):
            # The known-free route can end at a sensor-visible boundary.
            # Advance only a short live-LiDAR-verified step, then remap.
            yaw=wrap(math.atan2(intent.goal_xy[1]-pose.y,
                                intent.goal_xy[0]-pose.x)-pose.yaw)
            probe=lidar_unknown_step(pose,rr,aa,
                body_radius_px=(self.cfg.loaded_radius_px if carry else self.cfg.empty_radius_px),
                yaw_relative=yaw,max_step_px=18.)
            if probe is not None:waypoint=probe
        if waypoint is None and intent.phase in ('EXPLORE','CARRY_EXPLORE'):
            theta,gap=self._local_explore_direction(pose,rr,aa,
                                                     preferred_xy=intent.goal_xy)
            if gap>(self.cfg.loaded_radius_px if carry else self.cfg.empty_radius_px)+15:
                angle=pose.yaw+theta
                step_length=min(12.,max(0.,gap-40.))
                waypoint=(pose.x+step_length*math.cos(angle),pose.y+step_length*math.sin(angle))
        if waypoint is None and intent.phase=='CARRY':
            # A disconnected remembered center is a search lead, not a
            # reason to freeze with a live payload. Explore locally under the
            # loaded LiDAR clearance until the known-free graph reconnects.
            theta,gap=self._local_explore_direction(pose,rr,aa,
                                                     preferred_xy=intent.goal_xy)
            if gap>self.cfg.loaded_radius_px+18.:
                angle=pose.yaw+theta
                step_length=min(18.,max(0.,gap-40.))
                waypoint=(pose.x+step_length*math.cos(angle),
                          pose.y+step_length*math.sin(angle))
        if waypoint is None:
            # Missing route is NOT arrival; controlled halt/scan to avoid
            # destructive blind escapes. Heading changes in spacious regions.
            rotation=.14 if intent.phase=='EXPLORE' and np.nanmin(rr)>70 else 0.
            if step%10==0:
                self._trace.append({'step':step,'phase':intent.phase,'xy':[float(pose.x),float(pose.y)],
                    'goal':None if intent.goal_xy is None else list(map(float,intent.goal_xy)),
                    'carry':carry,'task':self.mem.my_target,'route_points':len(self.follow.route.points),
                    'reason':'NO_WAYPOINT'})
            return {'forward':0.,'lateral':0.,'rotation':rotation,'grasper':int(carry or intent.grasp)}
        decel=(self.cfg.loaded_decel_px_frame2 if carry else self.cfg.empty_decel_px_frame2)
        tolerance=(3. if direct_center or math.dist(pose.xy(),waypoint)<20. else 7.)
        desired=make_reference(pose.xy(),waypoint,speed_cap=intent.speed_cap,
                               brake_decel=decel,tolerance=tolerance)
        # Direct semantic near-target intentional contact is handled narrowly
        # by ModelAwareMotion; all other rays still bound motion.
        near_sem=intent.terminal_target if (direct_center or direct_live) else None
        command=self.motion.output(pose,st.velocity,desired,
                    lidar_ranges=rr,lidar_angles=aa,carrying=carry,
                    grasper=(carry or intent.grasp),desired_yaw=intent.yaw,
                    peers=near_peers,semantic_target=near_sem,
                    near_target=bool(near_sem is not None),
                    cargo_bearing=cargo_bearing,cargo_uncertainty=cargo_uncertainty)
        if step%10==0:
            self._trace.append({'step':step,'phase':intent.phase,'xy':[float(pose.x),float(pose.y)],
                'goal':None if intent.goal_xy is None else list(map(float,intent.goal_xy)),
                'waypoint':list(map(float,waypoint)),'carry':carry,'task':self.mem.my_target,
                'route_points':len(self.follow.route.points),'route_index':self.follow.index,
                'semantic_distance':None if near_sem is None else float(near_sem.distance),
                'reason':command.reason,'safe_speed':command.safe_speed,
                'command':[command.forward,command.lateral,command.rotation,command.grasper]})
        return {'forward':command.forward,'lateral':command.lateral,
                'rotation':command.rotation,'grasper':command.grasper}
