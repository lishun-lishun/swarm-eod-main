import numpy as np

from swarm_rescue.solutions.blue_pid_explore.predictive_brake import (
    BrakingMode,
    PredictiveBrakingController,
)


def test_nominal_action_passes_when_one_step_future_stop_is_safe():
    controller = PredictiveBrakingController()
    output = controller.compute(speed=8.0, nominal_thrust=0.5, remaining_distance=100.0)
    assert output.mode == BrakingMode.NORMAL
    assert output.safe_thrust == 0.5


def test_bisection_returns_boundary_feasible_maximum():
    controller = PredictiveBrakingController(bisection_iterations=25)
    output = controller.compute(speed=10.0, nominal_thrust=1.0, remaining_distance=60.0)
    assert output.mode == BrakingMode.PREDICTIVE_BRAKE
    assert output.safety_residual <= 1e-6
    assert controller.safety_residual(10.0, output.safe_thrust + 1e-5, 60.0, 0.0) > 0


def test_infeasible_state_is_explicit():
    output = PredictiveBrakingController().compute(
        speed=10.0, nominal_thrust=1.0, remaining_distance=20.0
    )
    assert output.mode == BrakingMode.INFEASIBLE_BRAKING
    assert not output.feasible
    assert output.safe_thrust == -1.0


def test_brake_release_is_rate_limited_but_restriction_is_immediate():
    controller = PredictiveBrakingController(recovery_rate=0.2)
    braking = controller.compute(speed=10.0, nominal_thrust=1.0, remaining_distance=60.0)
    released = controller.compute(speed=2.0, nominal_thrust=1.0, remaining_distance=100.0)
    assert released.safe_thrust <= braking.safe_thrust + 0.2 + 1e-12


def test_total_excursion_does_not_double_count_current_frame_peak():
    controller = PredictiveBrakingController()
    frame_end, _, frame_peak, future, total = controller.candidate_prediction(0.2, -1.0)
    assert np.isclose(total, max(frame_peak, frame_end + future))
    assert total <= frame_peak + future + 1e-12


def test_safety_residual_is_monotone_in_thrust():
    controller = PredictiveBrakingController()
    thrusts = np.linspace(-1.0, 1.0, 101)
    residuals = [controller.safety_residual(8.0, u, 50.0, 0.2) for u in thrusts]
    assert np.all(np.diff(residuals) >= -1e-10)
