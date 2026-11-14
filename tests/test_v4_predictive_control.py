"""Mathematical gates for the opt-in V4 controllers."""
import math

import numpy as np
import pytest

from swarm_rescue.solutions.blue_pid_explore.model_velocity_control import (
    EMPTY_MODEL, PlanarModel, PolePlacementVelocityController, SharedAxisRLS,
    ShadowCargoIdentifier,
    body_to_world, reference_velocity, unit_disk, world_to_body,
)
from swarm_rescue.solutions.blue_pid_explore.predictive_task_eta import (
    AssignmentHysteresis, CompletionTimeEstimator,
)


def test_world_body_round_trip():
    for heading in (-2.9, -0.4, 0.0, 1.7):
        command = np.array([0.31, -0.78])
        np.testing.assert_allclose(
            body_to_world(world_to_body(command, heading), heading), command,
            atol=1e-12)


def test_unit_disk_and_bad_measurement():
    assert np.linalg.norm(unit_disk([4.0, 3.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        unit_disk([math.nan, 0.0])


def test_pole_placement_converges_without_saturation():
    control = PolePlacementVelocityController(closed_loop_pole=0.75)
    velocity = np.zeros(2)
    target = np.array([3.0, -1.0])
    for _ in range(40):
        thrust = control.compute(target, velocity)
        assert np.linalg.norm(thrust) <= 1.0 + 1e-12
        velocity = EMPTY_MODEL.decay * velocity + EMPTY_MODEL.thrust_gain * thrust
    np.testing.assert_allclose(velocity, target, atol=0.01)


def test_reference_velocity_slows_at_target():
    far = reference_velocity((0, 0), (100, 0), max_speed=8,
                             braking_deceleration=0.5)
    near = reference_velocity((0, 0), (8, 0), max_speed=8,
                              braking_deceleration=0.5)
    stopped = reference_velocity((0, 0), (4, 0), max_speed=8,
                                 braking_deceleration=0.5)
    assert np.linalg.norm(far) > np.linalg.norm(near) > 0
    np.testing.assert_array_equal(stopped, np.zeros(2))


def test_eta_includes_return_and_failures():
    estimator = CompletionTimeEstimator()
    near = estimator.estimate(100, 200)
    far = estimator.estimate(100, 600)
    failed = estimator.estimate(100, 200, failure_count=2)
    unknown = estimator.estimate(100, None)
    assert near.total_frames < far.total_frames
    assert failed.total_frames > near.total_frames
    assert unknown.uncertain_home
    assert math.isinf(estimator.estimate(math.inf, 200).total_frames)


def test_assignment_hysteresis():
    gate = AssignmentHysteresis()
    assert not gate.should_switch(200, 170)
    assert gate.should_switch(200, 160)
    assert gate.should_switch(200, 190, incumbent_still_valid=False)


def test_rls_identifies_free_motion_only():
    fit = SharedAxisRLS(initial=PlanarModel(0.95, 0.19))
    rng = np.random.default_rng(27)
    velocity = np.zeros(2)
    for _ in range(600):
        thrust = rng.uniform(-0.7, 0.7, 2)
        next_velocity = 0.91 * velocity + 0.25 * thrust
        fit.observe(velocity, thrust, next_velocity, valid_free_motion=True)
        velocity = next_velocity
    model = fit.validated_model(min_updates=100, max_rmse=0.3)
    assert model is not None
    assert model.decay == pytest.approx(0.91, abs=0.01)
    assert model.thrust_gain == pytest.approx(0.25, abs=0.01)
    updates = fit.updates
    assert not fit.observe([math.nan, 0], [0, 1], [1, 0],
                           valid_free_motion=True)
    assert not fit.observe([1, 0], [0, 1], [1, 0],
                           valid_free_motion=False)
    assert fit.updates == updates


def test_cargo_shadow_aligns_last_action_and_skips_invalid_intervals():
    shadow = ShadowCargoIdentifier()
    rng = np.random.default_rng(28)
    velocity = np.zeros(2)
    for step in range(300):
        thrust = rng.uniform(-0.7, 0.7, 2)
        shadow.advance(step, velocity, thrust, eligible=True)
        velocity = 0.91 * velocity + 0.25 * thrust
    assert shadow.rls.updates >= 230
    assert shadow.rls.theta[0] == pytest.approx(0.91, abs=0.02)
    assert shadow.rls.theta[1] == pytest.approx(0.25, abs=0.02)
    updates = shadow.rls.updates
    shadow.advance(300, velocity, [0, 0], eligible=False)
    shadow.advance(302, velocity, [0, 0], eligible=True)
    shadow.advance(303, [math.nan, 0], [0, 0], eligible=True)
    assert shadow.rls.updates == updates
    assert shadow.diagnostics()["control_authorized"] is False
    assert shadow.diagnostics()["holdout_pairs"] > 0
    assert shadow.diagnostics()["holdout_rmse"] < 0.1
