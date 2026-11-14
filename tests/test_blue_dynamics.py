import numpy as np

from swarm_rescue.solutions.blue_pid_explore.dynamics import (
    DiscreteDynamicsModel,
    fit_discrete_dynamics,
)


def test_matrix_identification_recovers_known_model():
    decay = 0.95
    matrix = np.array([[0.57, 0.015], [-0.01, 0.59]])
    bias = np.array([0.002, -0.003])
    rng = np.random.default_rng(7)
    samples = []
    velocity = np.zeros(2)
    for step in range(160):
        command = rng.uniform(-1, 1, 2)
        norm = np.linalg.norm(command)
        if norm > 1:
            command /= norm
        following = decay * velocity + matrix @ command + bias
        samples.append({
            "experiment": f"segment_{step // 40}",
            "velocity_before": velocity.copy(),
            "command_world": command.copy(),
            "velocity_after": following.copy(),
        })
        velocity = following

    result = fit_discrete_dynamics(samples)
    assert np.isclose(result.model.velocity_decay, decay, atol=1e-10)
    assert np.allclose(result.model.input_matrix, matrix, atol=1e-10)
    assert np.allclose(result.model.bias, bias, atol=1e-10)
    assert result.validation_rmse < 1e-10


def test_stopping_trace_includes_delay_and_stops():
    model = DiscreteDynamicsModel(0.95, np.eye(2) * 0.57, np.zeros(2))
    speeds, distances = model.stopping_trace(8.0, control_delay_frames=1)
    assert speeds[1] == 7.6
    assert speeds[-1] <= 0.05
    assert np.all(np.diff(distances) >= 0)
    assert distances[-1] > 0
