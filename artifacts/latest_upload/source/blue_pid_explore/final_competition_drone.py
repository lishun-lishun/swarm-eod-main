"""Competition controller: proven SICT mission stack plus predictive safety."""

from __future__ import annotations

from collections import deque
import math
import os

import numpy as np

from .pose_fusion_drone import FusionRescueDrone
from .pose_sict_drone import EXPECTED_BOMBS, GOAL_STICKY, wrap
from .corridor_explorer import CorridorExplorer, CorridorExploreState
from .lidar_geometry import footprint_vertices_from_body
from .lidar_safety import LidarPredictiveSafetyLayer
from .oriented_footprint import OrientedFootprint
from .predictive_brake import BrakingMode, PredictiveBrakingController
from .velocity_controller import body_to_world, world_to_body


class FinalCompetitionDrone(FusionRescueDrone):
    """Keep the measured winning strategy and filter its final 2-D thrust.

    The legacy stack remains authoritative for mapping, gossip, allocation,
    pickup and cargo delivery.  The filter is deliberately downstream, so it
    cannot create a second mission state machine.  Semantic task targets are
    cut out of the LiDAR obstacle scan only at contact range; otherwise a bomb
    or the disposal centre would be indistinguishable from a wall and safe
    pickup/delivery would be impossible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictive_safety = LidarPredictiveSafetyLayer()
        self.corridor_brake = PredictiveBrakingController()
        self._safety_footprint = footprint_vertices_from_body(self.base)
        self.corridor_explorer = CorridorExplorer(
            OrientedFootprint(self._safety_footprint, margin=1.0),
            advance_speed=float(os.environ.get("CORRIDOR_ADVANCE_SPEED", "2.8")),
        )
        self.corridor_wall_margin = float(
            os.environ.get("CORRIDOR_WALL_MARGIN", "10.0"))
        specialist_text = os.environ.get("CORRIDOR_SPECIALISTS", "9").strip()
        corridor_specialists = (
            None if specialist_text == "*" else
            {int(value) for value in specialist_text.split(",") if value.strip()}
        )
        self.corridor_enabled = (
            os.environ.get("CORRIDOR_EXPLORER", "1") == "1"
            and (corridor_specialists is None
                 or int(self.identifier or 0) in corridor_specialists)
        )
        self.corridor_start_step = int(os.environ.get("CORRIDOR_START_STEP", "3000"))
        self.corridor_degraded_start = int(
            os.environ.get("CORRIDOR_DEGRADED_START", "-1"))
        self.corridor_owner_mod = int(
            os.environ.get("CORRIDOR_OWNER_MOD", "0"))
        self.degraded_pickup_slots = int(
            os.environ.get("DEGRADED_PICKUP_SLOTS", "0"))
        self.degraded_pickup_slot_frames = int(
            os.environ.get("DEGRADED_PICKUP_SLOT_FRAMES", "120"))
        self.corridor_trace_enabled = os.environ.get("CORRIDOR_TRACE", "0") == "1"
        self.carry_trace_enabled = os.environ.get("CARRY_TRACE", "0") == "1"
        self.carry_trace = deque(maxlen=360)
        self._trace_ring = deque(maxlen=30)
        self._trace_pending = []
        self.corridor_reversal_traces = []
        self._trace_last_world_command = None
        self._trace_last_goal = None
        self._corridor_control_state = CorridorExploreState.IDLE
        self._corridor_end_gap = math.inf
        self._corridor_goal_lock = False
        self._corridor_branch_goal = None
        self._last_blocked_reselect_step = -1000
        self._last_degraded_state = None
        self._gps_outage_frames = 0
        self._gps_recovery_frames = 60
        self._comm_outage_frames = 0
        self._comm_recovery_frames = 60
        self._stable_gps_degraded = False
        self._stable_comm_degraded = False
        self.safety_stats = {
            "filtered_frames": 0,
            "predictive_brake_frames": 0,
            "infeasible_frames": 0,
        }

    @staticmethod
    def _sector_min(distances, angles, center, half=0.18):
        d = np.asarray(distances, float)
        a = np.asarray(angles, float)
        error = np.abs((a - center + math.pi) % (2.0 * math.pi) - math.pi)
        values = d[np.isfinite(d) & (d > 0.0) & (error <= half)]
        return None if values.size == 0 else round(float(np.min(values)), 2)

    def _trace_explore(self, command, lidar, angles, reason):
        if not self.corridor_trace_enabled or self.pose is None:
            return
        step = int(self.elapsed_timestep)
        world = body_to_world((float(command["forward"]), float(command["lateral"])),
                              self.heading)
        goal = None if self.goal is None else [round(float(v), 2) for v in self.goal]
        sample = {
            "step": step,
            "mode": str(self.state),
            "reason": str(reason),
            "pose": [round(float(v), 2) for v in self.pose],
            "heading": round(float(self.heading), 4),
            "velocity": [round(float(v), 3) for v in self.vel],
            "command_world": [round(float(v), 3) for v in world],
            "goal": goal,
            "path_points": int(len(self.path)),
            "path_index": int(getattr(self, "path_i", 0)),
            "interest_left": None if self.interest_left is None else int(self.interest_left),
            "gps_valid": self.measured_gps_position() is not None,
            "lidar_min": round(float(np.nanmin(lidar)), 2),
            "lidar_front": self._sector_min(lidar, angles, 0.0),
            "lidar_left": self._sector_min(lidar, angles, math.pi / 2.0),
            "lidar_right": self._sector_min(lidar, angles, -math.pi / 2.0),
            "corridor_state": self.corridor_explorer.state.value,
        }
        for pending in list(self._trace_pending):
            pending["after"].append(sample)
            pending["remaining"] -= 1
            if pending["remaining"] <= 0:
                self._trace_pending.remove(pending)
        previous = self._trace_last_world_command
        previous_goal = self._trace_last_goal
        norm = float(np.linalg.norm(world))
        prev_norm = 0.0 if previous is None else float(np.linalg.norm(previous))
        reversed_motion = (
            previous is not None and norm > 0.18 and prev_norm > 0.18
            and float(np.dot(world, previous) / (norm * prev_norm)) < -0.35
        )
        goal_changed = previous_goal is not None and goal is not None and math.dist(
            previous_goal, goal) > 35.0
        if reversed_motion and len(self.corridor_reversal_traces) < 12:
            cause = ("GOAL_RESELECT" if goal_changed else
                     "RETURN" if self.state == "RETURN" else
                     "PATH_EMPTY" if not self.path else "LOCAL_CONTROL")
            record = {
                "drone": int(self.me), "step": step, "cause": cause,
                "before": list(self._trace_ring), "after": [sample], "remaining": 30,
            }
            self.corridor_reversal_traces.append(record)
            self._trace_pending.append(record)
        self._trace_ring.append(sample)
        self._trace_last_world_command = world.copy()
        self._trace_last_goal = None if goal is None else tuple(goal)

    def _corridor_travel_heading(self):
        target = self.goal
        if self.path:
            index = min(len(self.path) - 1, max(0, int(getattr(self, "path_i", 0))) + 3)
            target = self.path[index]
        if target is None or math.dist(self.pose, target) < 2.0:
            speed = math.hypot(*self.vel)
            return self.heading if speed < 0.2 else math.atan2(self.vel[1], self.vel[0])
        return math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])

    def _fetch(self, step, cmd, lidar, angles):
        self.corridor_explorer.pause()
        self._corridor_control_state = CorridorExploreState.PAUSED
        return super()._fetch(step, cmd, lidar, angles)

    def _carrier_slot_free(self):
        if not super()._carrier_slot_free():
            return False
        slots = self.degraded_pickup_slots
        if (slots <= 1
                or not (self._stable_gps_degraded
                        or self._stable_comm_degraded)):
            return True
        phase = (int(self.elapsed_timestep)
                 // max(1, self.degraded_pickup_slot_frames)) % slots
        return self.me % slots == phase

    def _carry(self, step, cmd, lidar, angles):
        self.corridor_explorer.pause()
        self._corridor_control_state = CorridorExploreState.PAUSED
        command = super()._carry(step, cmd, lidar, angles)
        if self.carry_trace_enabled and self.pose is not None:
            world = body_to_world(
                (float(command["forward"]), float(command["lateral"])), self.heading,
            )
            waypoint = None
            if self.path:
                index = min(len(self.path) - 1, max(0, int(getattr(self, "path_i", 0))))
                waypoint = [round(float(v), 2) for v in self.path[index]]
            center_cost = None
            if self.center_cum is not None:
                start = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
                if start is not None:
                    center_cost = round(float(self.center_cum[start]) * 10.0, 2)
            self.carry_trace.append({
                "step": int(step),
                "pose": [round(float(v), 2) for v in self.pose],
                "velocity": [round(float(v), 3) for v in self.vel],
                "command_world": [round(float(v), 3) for v in world],
                "path_goal": self.path_goal,
                "path_points": int(len(self.path)),
                "path_index": int(getattr(self, "path_i", 0)),
                "waypoint": waypoint,
                "center_cost": center_cost,
                "center_hits": int(len(self.center_hits)),
                "escape_until": int(self.escape_until),
                "escape_dir": None if self.escape_dir is None else [
                    round(float(v), 3) for v in self.escape_dir
                ],
                "lidar_min": None if lidar is None else round(float(np.nanmin(lidar)), 2),
                "lidar_front": None if lidar is None else self._sector_min(lidar, angles, 0.0),
                "lidar_back": None if lidar is None else self._sector_min(lidar, angles, math.pi),
                "lidar_left": None if lidar is None else self._sector_min(lidar, angles, math.pi / 2.0),
                "lidar_right": None if lidar is None else self._sector_min(lidar, angles, -math.pi / 2.0),
            })
        return command

    def _explore(self, step, cmd, lidar, angles):
        reason = "BASELINE"
        self._corridor_goal_lock = False
        bombs_missing = EXPECTED_BOMBS <= 0 or self.know.accounted() < EXPECTED_BOMBS
        team_idle = step - self.last_team_busy > 120
        existing_commitment = self.corridor_explorer.committed
        corridor_heading = self._corridor_travel_heading()
        raw_gps_degraded = self.measured_gps_position() is None
        raw_comm_degraded = self.communicator_is_disabled()
        if raw_gps_degraded:
            self._gps_outage_frames += 1
            self._gps_recovery_frames = 0
        else:
            self._gps_outage_frames = 0
            self._gps_recovery_frames += 1
        if raw_comm_degraded:
            self._comm_outage_frames += 1
            self._comm_recovery_frames = 0
        else:
            self._comm_outage_frames = 0
            self._comm_recovery_frames += 1
        if self._gps_outage_frames >= 8:
            self._stable_gps_degraded = True
        elif self._gps_recovery_frames >= 60:
            self._stable_gps_degraded = False
        if self._comm_outage_frames >= 4:
            self._stable_comm_degraded = True
        elif self._comm_recovery_frames >= 60:
            self._stable_comm_degraded = False
        gps_degraded = self._stable_gps_degraded
        comm_degraded = self._stable_comm_degraded
        degraded = gps_degraded or comm_degraded
        degraded_state = (gps_degraded, comm_degraded)
        if degraded_state != self._last_degraded_state:
            self._event(
                "zone_state", step, gps_degraded=bool(gps_degraded),
                comm_degraded=bool(comm_degraded))
            self._last_degraded_state = degraded_state
        degraded_trigger = (
            self.corridor_degraded_start >= 0
            and step >= self.corridor_degraded_start
            and degraded
        )
        if (degraded_trigger and not existing_commitment
                and self.corridor_owner_mod > 1):
            owner = int(math.floor(
                (wrap(corridor_heading) + math.pi)
                * self.corridor_owner_mod / (2.0 * math.pi)
            )) % self.corridor_owner_mod
            degraded_trigger = self.me % self.corridor_owner_mod == owner
        corridor_available = (
            self.corridor_enabled and bombs_missing
            and (existing_commitment
                 or degraded_trigger
                 or (step >= self.corridor_start_step and team_idle))
        )
        if not bombs_missing and self.corridor_explorer.state != CorridorExploreState.IDLE:
            self.corridor_explorer.reset()
            self._corridor_branch_goal = None
        if corridor_available and lidar is not None and angles is not None:
            previous_corridor_state = self.corridor_explorer.state
            if self.corridor_explorer.state == CorridorExploreState.PAUSED:
                self.corridor_explorer.resume(self.pose, step)
            output = self.corridor_explorer.update(
                step=step,
                position=self.pose,
                velocity=self.vel,
                heading=self.heading,
                travel_heading=corridor_heading,
                lidar_distances=lidar,
                lidar_angles=angles,
            )
            self._corridor_control_state = output.state
            self._corridor_end_gap = output.end_gap
            if (previous_corridor_state in (CorridorExploreState.IDLE,
                                            CorridorExploreState.COMPLETE)
                    and output.active):
                self.corridor_brake.reset()
            reason = output.reason
            if output.active:
                # A real bomb task bypasses _explore entirely and pauses this
                # commitment.  While active, frontier rescoring, scan_f and
                # interest_left are intentionally not consulted.
                if output.state == CorridorExploreState.ADVANCE:
                    # Preserve the semantic target and mature map path.  The
                    # corridor commitment only suppresses periodic frontier
                    # rescoring and scan_f invalidation; it does not replace
                    # the proven flight controller.
                    self._corridor_goal_lock = True
                    blocked_reselect = step - self.goal_eval_step > GOAL_STICKY
                    if self.goal is not None and self.goal_kind == "explore":
                        iy, ix = self.grid.cell(*self.goal)
                        scan_block = (
                            self.grid.scan_f[iy, ix] > 0
                            and self.grid.scan_f[iy, ix] > self.goal_step
                        )
                        blocked_reselect = blocked_reselect or scan_block
                    if (blocked_reselect
                            and step - self._last_blocked_reselect_step > GOAL_STICKY):
                        self.corridor_explorer.stats["midway_reversals_blocked"] += 1
                        self._last_blocked_reselect_step = step
                    command = super()._explore(step, cmd, lidar, angles)
                    self._trace_explore(command, lidar, angles, reason)
                    return command
                if (output.state == CorridorExploreState.BRANCH
                        and output.reason == "BRANCH_ENTER"):
                    if self._corridor_branch_goal is None:
                        branch_forward = np.array([
                            math.cos(output.branch_angle), math.sin(output.branch_angle)
                        ])
                        target = np.asarray(self.pose, float) + 70.0 * branch_forward
                        self._corridor_branch_goal = tuple(float(v) for v in target)
                        self.goal = self._corridor_branch_goal
                        self.goal_kind = "explore"
                        self.goal_deep = False
                        self.goal_step = step
                        self.goal_eval_step = step
                        self._refresh_dijkstra(step, force=True)
                        self._set_path_to(self.goal, step)
                        self.path_goal = "explore"
                    self._corridor_goal_lock = True
                    command = super()._explore(step, cmd, lidar, angles)
                    self._trace_explore(command, lidar, angles, reason)
                    return command
                self.state = "CORRIDOR"
                cmd["grasper"] = 0
                cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                    tuple(output.target_velocity), lidar, angles, False,
                    careful=output.state in (
                        CorridorExploreState.END_APPROACH,
                        CorridorExploreState.INSPECT,
                        CorridorExploreState.BRANCH,
                    ),
                    wall_margin=self.corridor_wall_margin,
                )
                if output.state != CorridorExploreState.INSPECT:
                    cmd["rotation"] = max(-0.55, min(
                        0.55, 0.8 * wrap(output.target_heading - self.heading)
                    ))
                self._trace_explore(cmd, lidar, angles, reason)
                return cmd
            if reason in ("CLOSED", "COMPLETE"):
                self.goal = None
                self.path = []
                self._corridor_branch_goal = None
        else:
            self._corridor_control_state = CorridorExploreState.IDLE
        command = super()._explore(step, cmd, lidar, angles)
        self._trace_explore(command, lidar, angles, reason)
        return command

    def _task_open_scan(self, distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
        scan = distances.copy()
        target = None
        half_angle = 0.0
        if self.grasped_bombs() and self.center_hits:
            nearest = min(self.center_hits, key=lambda hit: hit[2])
            if nearest[2] < 185.0:
                target = nearest[:2]
                half_angle = 0.72
        elif self.visible_bombs:
            position, distance = min(self.visible_bombs, key=lambda item: item[1])
            if distance < 58.0 and self.state == "FETCH":
                target = position
                half_angle = 0.48
        if target is None:
            return scan
        bearing = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        relative = np.abs(np.asarray([wrap(float(a) - (bearing - self.heading)) for a in angles]))
        scan[relative < half_angle] = max(230.0, float(np.max(scan)))
        return scan

    def control(self):
        command = super().control()
        if (not self.ready or self.pose is None or self.heading is None
                or int(self.elapsed_timestep) < 20):
            return command
        # The calibrated empty-airframe filter is used for its intended
        # regime: high-speed scouting.  Fetch and cargo dynamics are different
        # (the bomb is a swinging second body), and the historical controller's
        # measured load-specific braking / semantic contact logic is the valid
        # model there.  Applying the empty model to those modes reduced the
        # same-layout result from 10/10 to 3/10 despite higher health.
        corridor_mode = self.state == "CORRIDOR"
        if (self.state != "EXPLORE" and not corridor_mode) or self.grasped_bombs():
            return command
        distances = self.lidar_values()
        angles = self.lidar_rays_angles()
        if distances is None or angles is None:
            return command
        # The baseline already brakes from its cargo/empty identified models.
        # Invoke the costlier nonlinear guard only as the last emergency layer
        # for a fast scout with an actually imminent return.  Earlier use at
        # the full stopping horizon was safe but changed frontier arrival
        # timing enough to lose one late bomb on the fixed 2000-step budget.
        finite = np.asarray(distances, float)
        finite = finite[np.isfinite(finite)]
        if corridor_mode and self._corridor_control_state not in (
                CorridorExploreState.END_APPROACH,
                CorridorExploreState.INSPECT):
            return command
        if corridor_mode:
            # Endpoint braking is a single half-plane normal to the end wall.
            # Solving it as a scalar preserves the validated nonlinear
            # stopping model without inventing mutually opposed constraints
            # from the two corridor side walls.
            axis = np.array([
                math.cos(self.corridor_explorer.direction),
                math.sin(self.corridor_explorer.direction),
            ])
            nominal_world = body_to_world(
                (float(command["forward"]), float(command["lateral"])), self.heading,
            )
            nominal_normal = float(nominal_world @ axis)
            robust_margin = (
                self.corridor_explorer.geometry_margin
                + self.corridor_explorer.sigma_gain * self.corridor_explorer.lidar_sigma
            )
            brake = self.corridor_brake.compute(
                speed=max(0.0, float(np.asarray(self.vel) @ axis)),
                nominal_thrust=nominal_normal,
                remaining_distance=float(self._corridor_end_gap),
                safety_margin=robust_margin,
                velocity_uncertainty=0.35,
                velocity_confidence=1.0,
            )
            tangent = nominal_world - nominal_normal * axis
            tangent_limit = math.sqrt(max(0.0, 1.0 - brake.safe_thrust ** 2))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > tangent_limit and tangent_norm > 1e-12:
                tangent *= tangent_limit / tangent_norm
            # Preserve the safety-critical normal component and fit the
            # lateral component into the remaining unit-circle budget.
            safe_world = brake.safe_thrust * axis + tangent
            body = world_to_body(safe_world, self.heading)
            command["forward"] = float(body[0])
            command["lateral"] = float(body[1])
            self.safety_stats["filtered_frames"] += 1
            if brake.mode == BrakingMode.PREDICTIVE_BRAKE:
                self.safety_stats["predictive_brake_frames"] += 1
            elif brake.mode == BrakingMode.INFEASIBLE_BRAKING:
                self.safety_stats["infeasible_frames"] += 1
            return command
        # Corridor commitment by itself is not a near-wall condition.  Running
        # the joint half-plane guard against both side walls for every frame of
        # a narrow corridor makes the empty-airframe stopping model report
        # mutually infeasible constraints and can command needless reversals.
        # Keep the guard at *all* speeds in corridor mode, but only once an
        # obstacle is actually close.  This preserves the required low-speed
        # end inspection without filtering ordinary corridor cruise.
        close_obstacle = finite.size > 0 and float(np.min(finite)) < 30.0
        if (not close_obstacle or
                (not corridor_mode and float(np.linalg.norm(self.vel)) < 5.0)):
            return command
        distances = self._task_open_scan(np.asarray(distances, float), np.asarray(angles, float))
        nominal_world = body_to_world(
            (float(command["forward"]), float(command["lateral"])), self.heading,
        )
        output = self.predictive_safety.compute(
            lidar_distances=distances,
            lidar_angles=angles,
            estimated_position=self.pose,
            estimated_heading=self.heading,
            estimated_velocity=self.vel,
            nominal_thrust=nominal_world,
            footprint_vertices=self._safety_footprint,
            velocity_sigma=0.35,
            position_sigma=0.0,
        )
        body = world_to_body(output.command_world, self.heading)
        command["forward"] = float(body[0])
        command["lateral"] = float(body[1])
        self.safety_stats["filtered_frames"] += 1
        status = output.joint.status.value
        if status == "PREDICTIVE_BRAKE":
            self.safety_stats["predictive_brake_frames"] += 1
        elif status == "INFEASIBLE_BRAKING":
            self.safety_stats["infeasible_frames"] += 1
        return command

