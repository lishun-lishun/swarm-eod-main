"""Closed-form and recursive stopping distances for the identified plant."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamics import (
    INPUT_GAIN,
    POSITION_INPUT_GAIN,
    POSITION_VELOCITY_GAIN,
    VELOCITY_DECAY,
)


@dataclass(frozen=True)
class StoppingResult:
    distance: float
    frames: int
    final_velocity: float
    final_position: float
    furthest_frame: int


class NonlinearStoppingModel:
    """Maximum forward excursion under full reverse thrust."""

    def __init__(
        self,
        *,
        velocity_decay: float = VELOCITY_DECAY,
        input_gain: float = INPUT_GAIN,
        position_velocity_gain: float = POSITION_VELOCITY_GAIN,
        position_input_gain: float = POSITION_INPUT_GAIN,
        physics_substeps: int = 10,
    ):
        self.velocity_decay = float(velocity_decay)
        self.input_gain = float(input_gain)
        self.position_velocity_gain = float(position_velocity_gain)
        self.position_input_gain = float(position_input_gain)
        self.physics_substeps = int(physics_substeps)
        if not 0.0 < self.velocity_decay < 1.0 or self.input_gain <= 0:
            raise ValueError("invalid stable dynamics coefficients")

    def stopping_frames(self, speed: float) -> int:
        speed = max(0.0, float(speed))
        if speed <= 1e-14:
            return 0
        ratio = self.input_gain / (
            self.input_gain + (1.0 - self.velocity_decay) * speed
        )
        estimate = int(math.ceil(math.log(ratio) / math.log(self.velocity_decay) - 1e-12))
        estimate = max(1, estimate)
        # Verify the first crossing recursively to remove logarithm/ceil edge
        # ambiguity at speeds that cross almost exactly on a frame boundary.
        velocity = speed
        for frame in range(1, estimate + 3):
            velocity = self.velocity_decay * velocity - self.input_gain
            if velocity <= 0.0:
                return frame
        raise RuntimeError("failed to verify stopping frame")

    def velocity_closed_form(self, speed: float, frame: int) -> float:
        if frame < 0:
            raise ValueError("frame must be non-negative")
        decay = self.velocity_decay**frame
        return (
            decay * float(speed)
            - self.input_gain * (1.0 - decay) / (1.0 - self.velocity_decay)
        )

    def displacement_closed_form(self, speed: float, frames: int) -> float:
        if frames <= 0:
            return 0.0
        decay_sum = (1.0 - self.velocity_decay**frames) / (1.0 - self.velocity_decay)
        velocity_sum = (
            (float(speed) + self.input_gain / (1.0 - self.velocity_decay)) * decay_sum
            - self.input_gain * frames / (1.0 - self.velocity_decay)
        )
        return self.position_velocity_gain * velocity_sum - self.position_input_gain * frames

    def stopping_recursive(self, speed: float) -> StoppingResult:
        velocity = max(0.0, float(speed))
        if velocity <= 0.0:
            return StoppingResult(0.0, 0, 0.0, 0.0, 0)
        position = 0.0
        furthest = 0.0
        furthest_frame = 0
        frames = 0
        while velocity > 0.0 and frames < 1000:
            position += self.position_velocity_gain * velocity - self.position_input_gain
            frames += 1
            if position > furthest:
                furthest = position
                furthest_frame = frames
            velocity = self.velocity_decay * velocity - self.input_gain
        if frames >= 1000:
            raise RuntimeError("stopping recursion failed to terminate")
        return StoppingResult(furthest, frames, velocity, position, furthest_frame)

    def stopping_analytic(self, speed: float) -> StoppingResult:
        speed = max(0.0, float(speed))
        frames = self.stopping_frames(speed)
        if frames == 0:
            return StoppingResult(0.0, 0, 0.0, 0.0, 0)
        displacements = [self.displacement_closed_form(speed, j) for j in range(frames + 1)]
        furthest_frame = int(np.argmax(displacements))
        return StoppingResult(
            float(displacements[furthest_frame]),
            frames,
            self.velocity_closed_form(speed, frames),
            float(displacements[-1]),
            furthest_frame,
        )

    def stopping_distance(self, speed: float) -> float:
        """Fast microstep-exact distance used in the real-time constraint."""
        velocity = max(0.0, float(speed))
        if velocity <= 0.0:
            return 0.0
        position = 0.0
        furthest = 0.0
        dt = 1.0 / self.physics_substeps
        micro_decay = self.velocity_decay**dt
        impulse = self.input_gain / (micro_decay ** (self.physics_substeps - 1))
        frames = 0
        while velocity > 0.0 and frames < 1000:
            for microstep in range(self.physics_substeps):
                position += dt * velocity
                if position > furthest:
                    furthest = position
                velocity = micro_decay * velocity + (-impulse if microstep == 0 else 0.0)
            frames += 1
        if frames >= 1000:
            raise RuntimeError("fast stopping recursion failed to terminate")
        return furthest

    def stopping_microstep(self, speed: float) -> StoppingResult:
        """Maximum excursion including all physics microsteps."""
        velocity = max(0.0, float(speed))
        if velocity <= 0.0:
            return StoppingResult(0.0, 0, 0.0, 0.0, 0)
        position = 0.0
        furthest = 0.0
        furthest_frame = 0
        frames = 0
        while velocity > 0.0 and frames < 1000:
            local_positions, local_velocities = self.frame_microstep_trace(velocity, -1.0)
            local_max = position + float(np.max(local_positions))
            if local_max > furthest:
                furthest = local_max
                furthest_frame = frames + 1
            position += float(local_positions[-1])
            velocity = float(local_velocities[-1])
            frames += 1
        if frames >= 1000:
            raise RuntimeError("microstep stopping recursion failed to terminate")
        return StoppingResult(furthest, frames, velocity, position, furthest_frame)

    def frame_microstep_trace(self, speed: float, thrust: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact free-flight microstep recurrence used by official Pymunk.

        Pymunk advances position with the incoming velocity, applies damping
        and the force in the first 0.1 step, then clears the force.  This
        recurrence reproduces alpha=0.97728842 and beta=0.52907993.
        """

        thrust = float(np.clip(thrust, -1.0, 1.0))
        velocity = float(speed)
        position = 0.0
        positions = [position]
        velocities = [velocity]
        dt = 1.0 / self.physics_substeps
        micro_decay = self.velocity_decay**dt
        first_impulse = (self.input_gain / (micro_decay ** (self.physics_substeps - 1)))
        for index in range(self.physics_substeps):
            position += dt * velocity
            velocity = micro_decay * velocity + (first_impulse * thrust if index == 0 else 0.0)
            positions.append(position)
            velocities.append(velocity)
        return np.asarray(positions), np.asarray(velocities)

    def frame_max_displacement(self, speed: float, thrust: float) -> float:
        velocity = float(speed)
        position = 0.0
        furthest = 0.0
        thrust = float(np.clip(thrust, -1.0, 1.0))
        dt = 1.0 / self.physics_substeps
        micro_decay = self.velocity_decay**dt
        impulse = self.input_gain / (micro_decay ** (self.physics_substeps - 1))
        for microstep in range(self.physics_substeps):
            position += dt * velocity
            furthest = max(furthest, position)
            velocity = micro_decay * velocity + (impulse * thrust if microstep == 0 else 0.0)
        return furthest
