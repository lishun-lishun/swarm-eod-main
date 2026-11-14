import numpy as np

from swarm_rescue.solutions.blue_pid_explore.velocity_controller import (
    InverseDynamicsVelocityController,
    SteadyStateFeedbackVelocityController,
    body_to_world,
    world_to_body,
)


def test_inverse_controller_hits_unsaturated_next_velocity():
    controller = InverseDynamicsVelocityController()
    current = np.array([2.0, -1.0])
    target = np.array([2.2, -0.8])
    output = controller.compute(target, current)
    following = 0.95 * current + 0.5729312289729012 * output.command_world
    assert not output.saturated
    assert np.allclose(following, target)


def test_feedback_controller_has_requested_closed_loop_pole():
    controller = SteadyStateFeedbackVelocityController(closed_loop_pole=0.6)
    # Keep the command below saturation so the designed linear pole applies.
    target = np.array([4.0, 0.0])
    current = np.array([3.5, 0.0])
    output = controller.compute(target, current)
    following = 0.95 * current + 0.5729312289729012 * output.command_world
    assert np.isclose(target[0] - following[0], 0.6 * (target[0] - current[0]))


def test_world_body_transform_round_trip_and_circle_saturation():
    world = np.array([0.7, -0.4])
    heading = 1.2
    assert np.allclose(body_to_world(world_to_body(world, heading), heading), world)
    output = InverseDynamicsVelocityController().compute([10.0, 10.0], [0.0, 0.0])
    assert output.saturated
    assert np.isclose(np.linalg.norm(output.command_world), 1.0)
