import numpy as np

from swarm_rescue.solutions.blue_pid_explore.nonlinear_braking import NonlinearStoppingModel


def test_recursive_and_analytic_stopping_models_agree_and_are_monotone():
    model = NonlinearStoppingModel()
    speeds = np.linspace(0.0, 11.46, 250)
    distances = []
    for speed in speeds:
        recursive = model.stopping_recursive(speed)
        analytic = model.stopping_analytic(speed)
        assert recursive.frames == analytic.frames
        assert np.isclose(recursive.distance, analytic.distance, atol=1e-10)
        assert np.isclose(recursive.final_position, analytic.final_position, atol=1e-10)
        distances.append(analytic.distance)
    assert distances[0] == 0.0
    assert np.all(np.diff(distances) >= -1e-12)


def test_microstep_recurrence_reproduces_identified_frame_model():
    model = NonlinearStoppingModel()
    for speed, thrust in [(10.0, -1.0), (0.2, -1.0), (3.0, 0.4)]:
        positions, velocities = model.frame_microstep_trace(speed, thrust)
        assert np.isclose(positions[-1], 0.97728842 * speed + 0.52907993 * thrust, atol=2e-8)
        assert np.isclose(velocities[-1], 0.95 * speed + 0.5729312289729012 * thrust, atol=2e-12)


def test_microstep_stopping_distance_is_never_below_frame_endpoint_distance():
    model = NonlinearStoppingModel()
    for speed in np.linspace(0.0, 11.46, 50):
        assert model.stopping_microstep(speed).distance + 1e-12 >= model.stopping_analytic(speed).distance


def test_stopping_frame_boundary_is_verified():
    model = NonlinearStoppingModel()
    # Pick v0 so v4 is exactly zero in exact arithmetic.
    speed = model.input_gain * (1.0 - model.velocity_decay**4) / (
        (1.0 - model.velocity_decay) * model.velocity_decay**4
    )
    assert model.stopping_frames(speed) == 4
