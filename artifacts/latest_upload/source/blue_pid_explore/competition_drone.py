"""Official 10-drone rescue entry integrating high-level strategy and new flight control."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData

from .bomb_tracker import BombStatus, BombTrack, BombTracker
from .communication import CommunicationMemory
from .disposal_traffic import DisposalReservation, DisposalTrafficManager
from .lidar_geometry import dynamic_drone_boundary, footprint_vertices_from_body
from .lidar_safety import LidarPredictiveSafetyLayer
from .localization import RobustLocalization
from .mission_manager import MissionManager, MissionState
from .navigation_manager import NavigationManager
from .oriented_footprint import OrientedFootprint, angle_delta
from .pickup_controller import PickupController
from .task_allocator import Claim, DistributedTaskAllocator, TaskBid
from .velocity_controller import SteadyStateFeedbackVelocityController, world_to_body


class BluePidExploreDrone(DroneAbstract):
    """One lawful controller instance; coordination happens only through radio."""

    def __init__(self, identifier: Optional[int] = None, misc_data: Optional[MiscData] = None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, display_lidar_graph=False, **kwargs)
        self.drone_id=int(identifier or 0)
        self.localization=RobustLocalization()
        self.velocity_controller=SteadyStateFeedbackVelocityController()
        self.safety=LidarPredictiveSafetyLayer()
        self.footprint_vertices=footprint_vertices_from_body(self.base)
        self.footprint=OrientedFootprint(self.footprint_vertices,margin=1.0)
        size=misc_data.size_area if misc_data and misc_data.size_area else (1800,1200)
        self.navigation=NavigationManager(size,self.footprint)
        self.bombs=BombTracker(); self.allocator=DistributedTaskAllocator(self.drone_id)
        self.pickup=PickupController(); self.traffic=DisposalTrafficManager(self.drone_id)
        self.mission=MissionManager(); self.comms=CommunicationMemory(self.drone_id)
        self.center: np.ndarray|None=None; self.current_claim: str|None=None
        self.last_estimate=None; self.last_localization=None; self.last_command=np.zeros(2)
        self.last_rotation=0.0; self.last_goal=None; self._was_carrying=False
        self._disposed_count=0; self._stuck=0; self._last_progress_position=None
        self._held_exploration_target=None; self._exploration_target_step=-1000
        self.stats={"collisions":0,"replans":0,"grasp_success":0,"disposed_confirmed":0,
                    "se2_plans":0,"blocked":0}

    def _serialize_tracks(self):
        values=sorted(self.bombs.tracks.values(),key=lambda b:b.observed_step,reverse=True)[:10]
        return tuple({"bomb_id":b.bomb_id,"position":b.position.tolist(),"status":b.status.value,
                      "version":b.version,"observed_step":b.observed_step,"owner":b.owner,
                      "lease_until":b.lease_until} for b in values)

    def _serialize_claims(self):
        return tuple(c.__dict__.copy() for c in self.allocator.claims.values())

    def define_message_for_all(self) -> Any:
        estimate=self.last_estimate
        if estimate is None:
            pos=(0.0,0.0); heading=0.0; velocity=(0.0,0.0)
        else:
            pos=tuple(map(float,estimate.position)); heading=float(estimate.heading)
            velocity=tuple(map(float,estimate.velocity))
        goal=None if self.last_goal is None else tuple(map(float,self.last_goal))
        reservation=self.traffic.reservations.get(self.drone_id)
        return self.comms.make(step=int(self.elapsed_timestep),pose=(pos[0],pos[1],heading),
            velocity=velocity,mission=self.mission.state.value,goal=goal,
            bomb_updates=self._serialize_tracks(),claims=self._serialize_claims(),
            disposal_reservation=None if reservation is None else reservation.__dict__.copy(),
            center=None if self.center is None else tuple(map(float,self.center)))

    def _receive(self):
        for _,payload in self.communicator.received_messages:
            message=self.comms.receive(payload)
            if message is None: continue
            tracks=[]
            for row in message.bomb_updates:
                try: tracks.append(BombTrack(row["bomb_id"],np.asarray(row["position"],float),
                    BombStatus(row["status"]),int(row["version"]),int(row["observed_step"]),
                    row.get("owner"),int(row.get("lease_until",0))))
                except (KeyError,ValueError,TypeError): pass
            self.bombs.merge(tracks)
            claims=[]
            for row in message.claims:
                try: claims.append(Claim(**row))
                except TypeError: pass
            self.allocator.merge(claims,int(self.elapsed_timestep))
            if message.disposal_reservation:
                try:self.traffic.merge([DisposalReservation(**message.disposal_reservation)],int(self.elapsed_timestep))
                except TypeError:pass
            if self.center is None and message.center is not None:self.center=np.asarray(message.center,float)

    def _perceive_semantic(self,estimate):
        values=self.semantic_values() or []
        semantic_angles=self.semantic().ray_angles
        for detection in values:
            if not np.isfinite(detection.distance) or not np.isfinite(detection.angle):continue
            bearing=estimate.heading+float(detection.angle)
            point=estimate.position+float(detection.distance)*np.array([math.cos(bearing),math.sin(bearing)])
            if detection.entity_type==DroneSemanticSensor.TypeEntity.BOMB and not detection.grasped:
                self.bombs.observe(point,int(self.elapsed_timestep))
            elif detection.entity_type==DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER:
                self.center=point if self.center is None else 0.85*self.center+0.15*point
        return semantic_angles

    def _choose_task(self,estimate):
        if self.grasped_bombs():return
        self.allocator.expire(int(self.elapsed_timestep))
        available=[b for b in self.bombs.tracks.values() if b.status in (BombStatus.DISCOVERED,BombStatus.LOST)
                   and (b.bomb_id not in self.allocator.claims or self.allocator.claims[b.bomb_id].drone_id==self.drone_id)]
        if not available:return
        def bid(track):
            arrival=np.linalg.norm(track.position-estimate.position)/6.0
            return_dist=0.0 if self.center is None else np.linalg.norm(track.position-self.center)/4.0
            return TaskBid(track.bomb_id,self.drone_id,arrival,12.0,return_dist,0.0,int(self.elapsed_timestep))
        best=min((bid(b) for b in available),key=lambda x:(x.total_time,x.bomb_id))
        claim=self.allocator.claim(best,int(self.elapsed_timestep)); self.current_claim=claim.bomb_id
        track=self.bombs.tracks[claim.bomb_id]
        self.bombs.update_status(track.bomb_id,BombStatus.CLAIMED,int(self.elapsed_timestep),
                                 owner=self.drone_id,lease_until=claim.expires)
        self.mission.assign_bomb(track.bomb_id,track.position); self.pickup.reset()

    def _exploration_target(self,estimate):
        # Frontier target holding is essential both for navigation stability
        # and for the 180 s wall-clock budget.  Do not rerun a full grid search
        # merely because one more LiDAR cell changed this frame.
        if (self._held_exploration_target is not None
                and int(self.elapsed_timestep)-self._exploration_target_step < 120
                and np.linalg.norm(estimate.position-self._held_exploration_target)>15.0):
            return self._held_exploration_target.copy()
        peers=[m.goal for m in self.comms.peers.values() if m.goal is not None and m.mission==MissionState.EXPLORE.value]
        frontier=self.navigation.choose_frontier(estimate.position,peers)
        if frontier is not None:
            self.navigation.frontiers.mark_selected(frontier.representative)
            self._held_exploration_target=self.navigation.map.grid_to_world(frontier.representative)
            self._exploration_target_step=int(self.elapsed_timestep)
            return self._held_exploration_target.copy()
        # Deterministic legal fallback over the known map extent.
        width,height=self.size_area if self.size_area else (1600,1100)
        angle=2*math.pi*((self.drone_id*0.61803398875+(self.elapsed_timestep//300)*0.17)%1.0)
        radius=0.34*min(width,height)
        self._held_exploration_target=np.array([radius*math.cos(angle),radius*math.sin(angle)])
        self._exploration_target_step=int(self.elapsed_timestep)
        return self._held_exploration_target.copy()

    def _dynamic_boundaries(self,estimate):
        result=[]
        for peer in self.comms.peers.values():
            age=int(self.elapsed_timestep)-peer.step
            if age>20:continue
            other_pos=np.asarray(peer.pose[:2],float)+age*np.asarray(peer.velocity,float)
            if np.linalg.norm(other_pos-estimate.position)>180:continue
            result.append(dynamic_drone_boundary(estimate.position,other_pos,peer.velocity,
                self.footprint_vertices,self.footprint_vertices,estimate.heading,peer.pose[2],geometry_error=12.0))
        return result

    def control(self)->CommandsDict:
        self._receive()
        loc=self.localization.update(odometry=self.odometer_values(),compass=self.compass_values(),
                                     gps_position=self.gps_values())
        estimate=loc.estimate; self.last_estimate=estimate; self.last_localization=loc
        semantic_angles=self._perceive_semantic(estimate)
        # Geometry safety still consumes every LiDAR frame below.  Mapping is
        # deliberately staggered at 1/3 rate: consecutive 10-ray scans overlap
        # heavily, while updating ten private grids every control frame spent
        # most of the official wall-clock budget without adding useful cells.
        step = int(self.elapsed_timestep)
        if step < 3 or step % 3 == self.drone_id % 3:
            self.navigation.update_sensors(estimate.position,estimate.heading,self.lidar_values(),
                                           self.lidar_rays_angles(),semantic_angles)
        carrying=bool(self.grasped_bombs())
        if carrying and not self._was_carrying:
            self.stats["grasp_success"]+=1
            if self.current_claim in self.bombs.tracks:
                self.bombs.update_status(self.current_claim,BombStatus.PICKED,int(self.elapsed_timestep),owner=self.drone_id)
        disposed_confirmed=self._was_carrying and not carrying and self.center is not None and np.linalg.norm(estimate.position-self.center)<100
        if disposed_confirmed:
            self.stats["disposed_confirmed"]+=1
            if self.current_claim in self.bombs.tracks:self.bombs.update_status(self.current_claim,BombStatus.DISPOSED,int(self.elapsed_timestep))
            self.current_claim=None
        self._was_carrying=carrying
        if self.mission.state==MissionState.EXPLORE:self._choose_task(estimate)
        exploration=self._exploration_target(estimate) if self.mission.state==MissionState.EXPLORE else None
        target=self.mission.target
        near=target is not None and np.linalg.norm(estimate.position-target)<12
        admitted=True
        if carrying and self.center is not None and np.linalg.norm(estimate.position-self.center)<160:
            if self.drone_id not in self.traffic.reservations:self.traffic.reserve(int(self.elapsed_timestep),health=self.drone_health)
            admitted=self.traffic.admitted(int(self.elapsed_timestep))
        goal=self.mission.update(near_target=bool(near),grasped=carrying,disposed_confirmed=disposed_confirmed,
            disposal_target=self.center,exploration_target=exploration,queue_admitted=admitted)
        self.last_goal=goal.target
        grasper=1 if carrying else 0
        if goal.state==MissionState.PICKUP and goal.target is not None:
            pickup=self.pickup.compute(estimate.position,estimate.velocity,goal.target,grasped=carrying)
            target_velocity=pickup.target_velocity; grasper=pickup.grasper
            target_heading=math.atan2(target_velocity[1],target_velocity[0]) if np.linalg.norm(target_velocity)>0.1 else estimate.heading
        elif goal.state==MissionState.QUEUE:
            target_velocity=np.zeros(2);target_heading=estimate.heading;grasper=1
        elif goal.target is not None:
            nav=self.navigation.navigate(estimate.position,estimate.velocity,estimate.heading,goal.target,
                                         int(self.elapsed_timestep),force_replan=False)
            target_velocity=nav.target_velocity;target_heading=nav.target_heading
            self.stats["se2_plans"]+=int(nav.used_se2);self.stats["blocked"]+=int(nav.replan_required)
            if carrying:grasper=1
        else:
            target_velocity=np.zeros(2);target_heading=estimate.heading
        # Degraded localization lowers risky speed but never freezes exploration.
        speed_limit=3.0 if loc.degraded else (4.5 if carrying else 8.0)
        speed=float(np.linalg.norm(target_velocity))
        if speed>speed_limit:target_velocity=target_velocity*speed_limit/speed
        nominal=self.velocity_controller.compute(target_velocity,estimate.velocity).command_world
        safe=self.safety.compute(lidar_distances=self.lidar_values(),lidar_angles=self.lidar_rays_angles(),
            estimated_position=estimate.position,estimated_heading=estimate.heading,
            estimated_velocity=estimate.velocity,nominal_thrust=nominal,
            footprint_vertices=self.footprint_vertices,velocity_sigma=loc.velocity_sigma,
            position_sigma=min(8.0,loc.position_sigma),dynamic_boundaries=self._dynamic_boundaries(estimate))
        rotation=float(np.clip(1.2*angle_delta(target_heading,estimate.heading),-1.0,1.0))
        if self.navigation.used_se2:rotation=float(np.clip(rotation,-0.55,0.55))
        body=world_to_body(safe.command_world,estimate.heading)
        self.localization.set_previous_command(safe.command_world,rotation)
        self.last_command=safe.command_world;self.last_rotation=rotation
        return {"forward":float(body[0]),"lateral":float(body[1]),"rotation":rotation,"grasper":int(grasper)}

