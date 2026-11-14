"""Identification helpers for the per-control-frame translational model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


VELOCITY_DECAY = 0.95
INPUT_GAIN = 0.5729312289729012
POSITION_VELOCITY_GAIN = 0.97728842
POSITION_INPUT_GAIN = 0.52907993
MAX_STEADY_SPEED = INPUT_GAIN / (1.0 - VELOCITY_DECAY)


@dataclass(frozen=True)
class DiscreteDynamicsModel:
    """Model ``v[k+1] = lambda * v[k] + B @ u[k] + bias``.

    Velocities and positions use simulator pixels and one strategy control
    frame is the time unit.  ``input_matrix`` is always 2x2, including for an
    isotropic fit.
    """

    velocity_decay: float
    input_matrix: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.input_matrix, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("input_matrix must have shape (2, 2)")
        if bias.shape != (2,):
            raise ValueError("bias must have shape (2,)")
        object.__setattr__(self, "input_matrix", matrix)
        object.__setattr__(self, "bias", bias)

    def predict(self, velocity: Sequence[float], command_world: Sequence[float]) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float)
        command = np.asarray(command_world, dtype=float)
        return self.velocity_decay * velocity + self.input_matrix @ command + self.bias

    def stopping_trace(
        self,
        initial_speed: float,
        *,
        stop_threshold: float = 0.05,
        max_frames: int = 500,
        control_delay_frames: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return projected speeds and conservative frame-summed distances.

        Braking is along the negative initial velocity direction.  The first
        ``control_delay_frames`` use zero input to expose the latency margin
        required by the predictive safety layer.
        """

        if initial_speed < 0 or stop_threshold < 0:
            raise ValueError("speeds must be non-negative")
        direction = np.array([1.0, 0.0])
        velocity = initial_speed * direction
        speeds = [float(initial_speed)]
        distances = [0.0]
        distance = 0.0
        for frame in range(max_frames):
            command = np.zeros(2) if frame < control_delay_frames else -direction
            next_velocity = self.predict(velocity, command)
            projected = max(0.0, float(next_velocity @ direction))
            # Trapezoidal integration is a better approximation of the
            # within-frame displacement than using only either endpoint.
            distance += 0.5 * (max(0.0, float(velocity @ direction)) + projected)
            velocity = next_velocity
            speeds.append(projected)
            distances.append(distance)
            if projected <= stop_threshold:
                break
        return np.asarray(speeds), np.asarray(distances)


@dataclass(frozen=True)
class BrakingPrediction:
    distance: float
    frames: int
    final_speed: float


def predict_forward_braking_distance(
    speed: float,
    *,
    velocity_decay: float = VELOCITY_DECAY,
    input_gain: float = INPUT_GAIN,
    position_velocity_gain: float = POSITION_VELOCITY_GAIN,
    position_input_gain: float = POSITION_INPUT_GAIN,
    control_delay_frames: int = 0,
    max_frames: int = 500,
) -> BrakingPrediction:
    """Predict maximum forward excursion under full reverse thrust.

    This uses the separately identified within-frame position increment model
    and returns the maximum excursion, not the final position after the first
    frame whose endpoint velocity is negative.
    """

    if speed < 0:
        raise ValueError("speed must be non-negative")
    velocity = float(speed)
    position = 0.0
    furthest = 0.0
    for frame in range(max_frames):
        command = 0.0 if frame < control_delay_frames else -1.0
        position += position_velocity_gain * velocity + position_input_gain * command
        furthest = max(furthest, position)
        velocity = velocity_decay * velocity + input_gain * command
        if frame >= control_delay_frames and velocity <= 0.0:
            return BrakingPrediction(furthest, frame + 1, velocity)
    raise RuntimeError("braking prediction did not stop within max_frames")


@dataclass(frozen=True)
class IdentificationResult:
    model: DiscreteDynamicsModel
    train_mae: float
    train_rmse: float
    validation_mae: float
    validation_rmse: float
    train_count: int
    validation_count: int


def _as_samples(samples: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(samples)
    if len(rows) < 8:
        raise ValueError("at least eight transition samples are required")
    previous = np.asarray([row["velocity_before"] for row in rows], dtype=float)
    commands = np.asarray([row["command_world"] for row in rows], dtype=float)
    following = np.asarray([row["velocity_after"] for row in rows], dtype=float)
    groups = np.asarray([row.get("experiment", "") for row in rows], dtype=object)
    if previous.shape[1:] != (2,) or commands.shape[1:] != (2,) or following.shape[1:] != (2,):
        raise ValueError("velocity and command fields must be two-dimensional")
    return previous, commands, following, groups


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = actual - predicted
    return float(np.mean(np.abs(residual))), float(np.sqrt(np.mean(residual**2)))


def fit_discrete_dynamics(
    samples: Iterable[dict],
    *,
    matrix_input: bool = True,
    validation_stride: int = 5,
) -> IdentificationResult:
    """Fit the model by least squares with deterministic grouped holdout.

    Every ``validation_stride``-th transition *within each experiment* is held
    out.  This prevents reset boundaries from becoming transitions and keeps
    all manoeuvre types represented in validation.
    """

    if validation_stride < 2:
        raise ValueError("validation_stride must be at least 2")
    previous, commands, following, groups = _as_samples(samples)
    validation = np.zeros(len(previous), dtype=bool)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        validation[indices[validation_stride - 1 :: validation_stride]] = True
    if not validation.any() or validation.all():
        raise ValueError("holdout split produced an empty partition")

    # A common scalar decay is deliberate: it matches the requested model and
    # makes anisotropy visible in B instead of hiding it in two decay values.
    if matrix_input:
        # x equation: [vx, ux, uy, 1]; y equation: [vy, ux, uy, 1].  Stack
        # both while sharing lambda, then solve once.
        n = len(previous)
        design = np.zeros((2 * n, 7), dtype=float)
        target = following.reshape(-1)
        design[0::2, 0] = previous[:, 0]
        design[1::2, 0] = previous[:, 1]
        design[0::2, 1:3] = commands
        design[1::2, 3:5] = commands
        design[0::2, 5] = 1.0
        design[1::2, 6] = 1.0
        train_rows = np.repeat(~validation, 2)
        params, *_ = np.linalg.lstsq(design[train_rows], target[train_rows], rcond=None)
        model = DiscreteDynamicsModel(
            velocity_decay=float(params[0]),
            input_matrix=np.array([[params[1], params[2]], [params[3], params[4]]]),
            bias=np.array([params[5], params[6]]),
        )
    else:
        # Shared lambda and shared scalar b across both axes.
        n = len(previous)
        design = np.zeros((2 * n, 4), dtype=float)
        target = following.reshape(-1)
        design[0::2] = np.column_stack((previous[:, 0], commands[:, 0], np.ones(n), np.zeros(n)))
        design[1::2] = np.column_stack((previous[:, 1], commands[:, 1], np.zeros(n), np.ones(n)))
        train_rows = np.repeat(~validation, 2)
        params, *_ = np.linalg.lstsq(design[train_rows], target[train_rows], rcond=None)
        model = DiscreteDynamicsModel(
            velocity_decay=float(params[0]),
            input_matrix=np.eye(2) * float(params[1]),
            bias=np.array([params[2], params[3]]),
        )

    predicted = np.asarray([model.predict(v, u) for v, u in zip(previous, commands)])
    train_mae, train_rmse = _metrics(following[~validation], predicted[~validation])
    val_mae, val_rmse = _metrics(following[validation], predicted[validation])
    return IdentificationResult(
        model=model,
        train_mae=train_mae,
        train_rmse=train_rmse,
        validation_mae=val_mae,
        validation_rmse=val_rmse,
        train_count=int((~validation).sum()),
        validation_count=int(validation.sum()),
    )
