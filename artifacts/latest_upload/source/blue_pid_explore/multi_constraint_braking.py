"""Joint two-dimensional predictive braking for multiple LiDAR boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Sequence

import numpy as np

from .dynamics import (
    INPUT_GAIN, POSITION_INPUT_GAIN, POSITION_VELOCITY_GAIN, VELOCITY_DECAY,
)
from .lidar_geometry import LidarBoundary
from .nonlinear_braking import NonlinearStoppingModel
from .velocity_controller import limit_unit_circle


class SafetyStatus(str, Enum):
    NORMAL = "NORMAL"
    PREDICTIVE_BRAKE = "PREDICTIVE_BRAKE"
    INFEASIBLE_BRAKING = "INFEASIBLE_BRAKING"


@dataclass(frozen=True)
class JointSafetyOutput:
    safe_thrust: np.ndarray
    nominal_thrust: np.ndarray
    status: SafetyStatus
    feasible: bool
    blocked: bool
    residuals: np.ndarray
    max_residual: float
    active_constraints: tuple[int, ...]
    candidate_count: int
    compute_time_ns: int
    predicted_trajectory: np.ndarray


class MultiConstraintBrakingSolver:
    """Small deterministic disk search with one shared future braking policy.

    After the candidate frame, every trajectory uses a single unit command
    opposite the full 2-D velocity.  Thus simultaneous wall constraints never
    assume that each normal independently receives full reverse thrust.
    """

    def __init__(
        self,
        *,
        radial_samples: int = 3,
        angular_samples: int = 16,
        max_future_frames: int = 80,
        physics_substeps: int = 10,
        recovery_rate: float = 0.2,
        blocked_frames: int = 5,
        fallback_disk_search: bool = False,
    ):
        self.radial_samples = int(radial_samples)
        self.angular_samples = int(angular_samples)
        self.max_future_frames = int(max_future_frames)
        self.physics_substeps = int(physics_substeps)
        self.recovery_rate = float(recovery_rate)
        self.blocked_frames = int(blocked_frames)
        self.fallback_disk_search = bool(fallback_disk_search)
        self._last_safe: np.ndarray | None = None
        self._infeasible_frames = 0
        self._model = NonlinearStoppingModel(physics_substeps=physics_substeps)

    def reset(self) -> None:
        self._last_safe = None
        self._infeasible_frames = 0

    def _frame(self, position: np.ndarray, velocity: np.ndarray, command: np.ndarray):
        dt = 1.0 / self.physics_substeps
        decay = VELOCITY_DECAY**dt
        impulse = INPUT_GAIN / (decay ** (self.physics_substeps - 1))
        samples = [position.copy()]
        for micro in range(self.physics_substeps):
            position = position + dt * velocity
            samples.append(position.copy())
            velocity = decay * velocity + (impulse * command if micro == 0 else 0.0)
        return position, velocity, samples

    def predict_shared_braking(self, velocity: Sequence[float], candidate: Sequence[float]):
        position = np.zeros(2, dtype=float)
        velocity = np.asarray(velocity, dtype=float).copy()
        command, _ = limit_unit_circle(candidate)
        position, velocity, samples = self._frame(position, velocity, command)
        # Isotropic identified dynamics plus a command exactly opposite the
        # *full 2-D velocity* preserve direction during every subsequent brake
        # frame.  The exact microstep stopping model therefore reduces the
        # remaining shared trajectory to one straight segment.  Its projection
        # on every boundary is monotone, so current 10 microsteps + the terminal
        # endpoint contain every possible maximum.  This is mathematically the
        # same joint policy as the recursive implementation, without its Python
        # frame/substep loop.
        speed = float(np.linalg.norm(velocity))
        if speed > 0.03:
            terminal = position + velocity / speed * self._model.stopping_distance(speed)
            samples.append(terminal)
        return np.asarray(samples), np.zeros(2)

    def residuals(
        self,
        velocity,
        candidate,
        boundaries: Sequence[LidarBoundary],
        *,
        velocity_sigma: float,
        position_sigma: float,
        k_v: float,
        k_p: float,
        base_margin: float,
        lidar_sigma_gain: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not boundaries:
            return np.empty(0), np.zeros((1, 2))
        velocity = np.asarray(velocity, dtype=float)
        candidate = np.asarray(candidate, dtype=float)
        shared_trajectory, _ = self.predict_shared_braking(velocity, candidate)
        worst = []
        for boundary in boundaries:
            n = boundary.normal
            # The actual candidate and shared future brake are validated once
            # in 2-D.  Per-boundary uncertainty then adds only the scalar
            # difference between v_n and robust v_n+, avoiding a second full
            # trajectory simulation and never treating the base command as
            # independently available along every normal.
            relative_velocity = velocity - (boundary.dynamic_velocity
                if boundary.dynamic_velocity is not None else 0.0)
            if boundary.dynamic_velocity is None:
                relative_trajectory = shared_trajectory
            else:
                # Moving peers require a relative-motion trajectory.  There
                # are normally only one or two such constraints, so retain
                # the exact shared 2-D prediction instead of folding peer
                # motion into a static clearance approximation.
                relative_trajectory, _ = self.predict_shared_braking(
                    relative_velocity, candidate,
                )
            excursion = float(np.max(relative_trajectory @ n))
            normal_speed = max(0.0, float(relative_velocity @ n))
            robust_speed = normal_speed + max(0.0, k_v * velocity_sigma)
            normal_thrust = float(candidate @ n)
            excursion += max(0.0, self._scalar_total(robust_speed, normal_thrust)
                             - self._scalar_total(normal_speed, normal_thrust))
            margin = (
                base_margin
                + lidar_sigma_gain * max(0.0, boundary.distance_sigma)
                + k_p * max(0.0, position_sigma)
                + max(0.0, boundary.geometry_error)
            )
            worst.append(excursion + margin - boundary.clearance)
        return np.asarray(worst), shared_trajectory

    def _scalar_total(self, speed: float, thrust: float) -> float:
        thrust = float(np.clip(thrust, -1.0, 1.0))
        next_velocity = VELOCITY_DECAY * float(speed) + INPUT_GAIN * thrust
        frame_end = POSITION_VELOCITY_GAIN * float(speed) + POSITION_INPUT_GAIN * thrust
        frame_peak = self._model.frame_max_displacement(speed, thrust)
        future = self._model.stopping_distance(max(0.0, next_velocity))
        return max(frame_peak, frame_end + future)

    def _candidates(self, nominal: np.ndarray, boundaries: Sequence[LidarBoundary]) -> list[np.ndarray]:
        nominal, _ = limit_unit_circle(nominal)
        values = [nominal, np.zeros(2)]
        radii = np.linspace(1.0 / self.radial_samples, 1.0, self.radial_samples)
        for radius in radii:
            for angle in np.linspace(-math.pi, math.pi, self.angular_samples, endpoint=False):
                values.append(radius * np.array([math.cos(angle), math.sin(angle)]))
        # Exact full braking and tangent candidates improve corner/wall quality.
        for boundary in boundaries:
            n, t = boundary.normal, boundary.tangent
            values.extend([-n, t, -t, nominal - max(0.0, float(nominal @ n)) * n])
        return values

    @staticmethod
    def _direct_candidates(nominal: np.ndarray, velocity: np.ndarray,
                           boundaries: Sequence[LidarBoundary]) -> list[np.ndarray]:
        """Constraint-informed candidates before the fallback disk search."""
        values = [np.zeros(2)]
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-9:
            values.append(-velocity / speed)
        normal_sum = np.sum([b.normal for b in boundaries], axis=0)
        normal_sum_norm = float(np.linalg.norm(normal_sum))
        if normal_sum_norm > 1e-9:
            values.append(-normal_sum / normal_sum_norm)
        for boundary in boundaries:
            n, t = boundary.normal, boundary.tangent
            tangent = float(np.clip(nominal @ t, -0.95, 0.95))
            normal = -math.sqrt(max(0.0, 1.0 - tangent * tangent))
            values.append(tangent * t + normal * n)
            values.append(-n)
        return values

    def compute(
        self,
        *,
        velocity: Sequence[float],
        nominal_thrust: Sequence[float],
        boundaries: Sequence[LidarBoundary],
        velocity_sigma: float = 0.0,
        position_sigma: float = 0.0,
        k_v: float = 1.0,
        k_p: float = 1.0,
        base_margin: float = 0.5,
        lidar_sigma_gain: float = 1.0,
    ) -> JointSafetyOutput:
        started = time.perf_counter_ns()
        nominal, _ = limit_unit_circle(nominal_thrust)
        if not boundaries:
            self._last_safe = nominal.copy()
            self._infeasible_frames = 0
            return JointSafetyOutput(nominal, nominal, SafetyStatus.NORMAL, True, False,
                                     np.empty(0), -math.inf, (), 1,
                                     time.perf_counter_ns() - started, np.zeros((1, 2)))
        # Most navigation frames are far from obstacles.  Validate the nominal
        # command once and avoid constructing/evaluating the disk grid when it
        # is already jointly safe.  Recovery hysteresis is still checked with
        # the exact same trajectory predicate.
        nominal_residuals, nominal_trajectory = self.residuals(
            velocity, nominal, boundaries, velocity_sigma=velocity_sigma,
            position_sigma=position_sigma, k_v=k_v, k_p=k_p,
            base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
        )
        if float(np.max(nominal_residuals)) <= 1e-9:
            safe = nominal.copy()
            residuals = nominal_residuals
            trajectory = nominal_trajectory
            status = SafetyStatus.NORMAL
            if self._last_safe is not None:
                delta = nominal - self._last_safe
                norm = float(np.linalg.norm(delta))
                if norm > self.recovery_rate:
                    limited = self._last_safe + self.recovery_rate * delta / norm
                    limited_res, limited_traj = self.residuals(
                        velocity, limited, boundaries, velocity_sigma=velocity_sigma,
                        position_sigma=position_sigma, k_v=k_v, k_p=k_p,
                        base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
                    )
                    if float(np.max(limited_res)) <= 1e-9:
                        safe, residuals, trajectory = limited, limited_res, limited_traj
                        status = SafetyStatus.PREDICTIVE_BRAKE
            self._last_safe = safe.copy()
            self._infeasible_frames = 0
            maximum = float(np.max(residuals))
            active = tuple(int(i) for i in np.flatnonzero(residuals >= maximum - 1e-5))
            return JointSafetyOutput(
                safe, nominal, status, True, False, residuals, maximum, active,
                1, time.perf_counter_ns() - started, trajectory,
            )
        velocity_array = np.asarray(velocity, dtype=float)
        direct = self._direct_candidates(nominal, velocity_array, boundaries)
        direct_feasible_raw = []
        direct_best_failure = None
        for candidate in direct:
            candidate, _ = limit_unit_circle(candidate)
            direct_res, direct_traj = self.residuals(
                velocity, candidate, boundaries, velocity_sigma=velocity_sigma,
                position_sigma=position_sigma, k_v=k_v, k_p=k_p,
                base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
            )
            direct_max = float(np.max(direct_res))
            item = (direct_max, candidate, direct_res, direct_traj)
            if direct_best_failure is None or direct_max < direct_best_failure[0]:
                direct_best_failure = item
            if direct_max <= 1e-9:
                direct_feasible_raw.append((float(np.sum((candidate - nominal) ** 2)),
                                            candidate, direct_res, direct_traj))
        direct_feasible = []
        if direct_feasible_raw:
            _, safe_point, safe_res, safe_traj = min(direct_feasible_raw, key=lambda item: item[0])
            unsafe_point = nominal
            # Refine only the best constraint-informed feasible ray.  Every
            # accepted midpoint is revalidated by the joint trajectory test.
            for _ in range(7):
                midpoint = 0.5 * (safe_point + unsafe_point)
                mid_res, mid_traj = self.residuals(
                    velocity, midpoint, boundaries, velocity_sigma=velocity_sigma,
                    position_sigma=position_sigma, k_v=k_v, k_p=k_p,
                    base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
                )
                if float(np.max(mid_res)) <= 1e-9:
                    safe_point, safe_res, safe_traj = midpoint, mid_res, mid_traj
                else:
                    unsafe_point = midpoint
            direct_feasible.append((float(np.sum((safe_point - nominal) ** 2)),
                                    safe_point, safe_res, safe_traj))
        # Only invoke the general disk search if the informed candidates found
        # no jointly feasible shared-braking trajectory.
        candidates = ([] if direct_feasible or not self.fallback_disk_search
                      else self._candidates(nominal, boundaries))
        feasible: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        best_failure = direct_best_failure
        for candidate in candidates:
            candidate, _ = limit_unit_circle(candidate)
            residuals, trajectory = self.residuals(
                velocity, candidate, boundaries, velocity_sigma=velocity_sigma,
                position_sigma=position_sigma, k_v=k_v, k_p=k_p,
                base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
            )
            max_residual = float(np.max(residuals))
            item = (max_residual, candidate, residuals, trajectory)
            if best_failure is None or max_residual < best_failure[0]:
                best_failure = item
            if max_residual <= 1e-9:
                feasible.append((float(np.sum((candidate - nominal) ** 2)), candidate, residuals, trajectory))
        feasible.extend(direct_feasible)
        if feasible:
            _, safe, residuals, trajectory = min(feasible, key=lambda item: item[0])
            # Rate-limit release only when the limited vector is itself safe.
            if self._last_safe is not None:
                delta = safe - self._last_safe
                norm = float(np.linalg.norm(delta))
                if norm > self.recovery_rate:
                    limited = self._last_safe + self.recovery_rate * delta / norm
                    limited_res, limited_traj = self.residuals(
                        velocity, limited, boundaries, velocity_sigma=velocity_sigma,
                        position_sigma=position_sigma, k_v=k_v, k_p=k_p,
                        base_margin=base_margin, lidar_sigma_gain=lidar_sigma_gain,
                    )
                    if float(np.max(limited_res)) <= 1e-9:
                        safe, residuals, trajectory = limited, limited_res, limited_traj
            status = SafetyStatus.NORMAL if np.linalg.norm(safe - nominal) <= 1e-6 else SafetyStatus.PREDICTIVE_BRAKE
            self._infeasible_frames = 0
            is_feasible = True
        else:
            _, safe, residuals, trajectory = best_failure
            status = SafetyStatus.INFEASIBLE_BRAKING
            self._infeasible_frames += 1
            is_feasible = False
        self._last_safe = safe.copy()
        maximum = float(np.max(residuals))
        active = tuple(int(i) for i in np.flatnonzero(residuals >= maximum - 1e-5))
        return JointSafetyOutput(
            safe, nominal, status, is_feasible,
            self._infeasible_frames >= self.blocked_frames,
            residuals, maximum, active, len(direct) + len(candidates),
            time.perf_counter_ns() - started, trajectory,
        )

