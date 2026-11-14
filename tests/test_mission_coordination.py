import numpy as np

from swarm_rescue.solutions.blue_pid_explore.bomb_tracker import BombStatus, BombTracker
from swarm_rescue.solutions.blue_pid_explore.disposal_traffic import DisposalTrafficManager
from swarm_rescue.solutions.blue_pid_explore.pickup_controller import PickupController, PickupState
from swarm_rescue.solutions.blue_pid_explore.task_allocator import DistributedTaskAllocator, TaskBid


def test_bomb_identity_merges_nearby_observations_and_tombstone_wins():
    tracker = BombTracker()
    a = tracker.observe([10, 10], 1); b = tracker.observe([13, 9], 2)
    assert a.bomb_id == b.bomb_id
    disposed = tracker.update_status(a.bomb_id, BombStatus.DISPOSED, 3)
    other = BombTracker(); other.merge([disposed])
    assert other.tracks[a.bomb_id].status == BombStatus.DISPOSED


def test_auction_uses_total_completion_time_not_nearest_only():
    bids = [TaskBid("b", 0, 2, 3, 30, 0, 1), TaskBid("b", 1, 8, 3, 5, 0, 1)]
    assert DistributedTaskAllocator.winner(bids).drone_id == 1


def test_claim_conflict_is_deterministic_and_expires():
    allocator = DistributedTaskAllocator(0, lease_steps=10)
    bids = [TaskBid("b", 2, 5, 1, 5, 0, 0), TaskBid("b", 1, 5, 1, 5, 0, 0)]
    claims = [DistributedTaskAllocator(i).claim(bid, 0) for i, bid in enumerate(bids)]
    allocator.merge(claims, 1)
    assert allocator.claims["b"].drone_id == 1
    allocator.expire(1000); assert "b" not in allocator.claims


def test_pickup_requires_actual_grasp_confirmation():
    controller = PickupController(max_retries=1)
    for _ in range(3):
        command = controller.compute([0, 0], [0, 0], [4, 0], grasped=False)
    assert command.state != PickupState.SUCCESS
    command = controller.compute([0, 0], [0, 0], [4, 0], grasped=True)
    assert command.state == PickupState.SUCCESS


def test_disposal_reservations_do_not_overlap():
    a = DisposalTrafficManager(0, slot_steps=20); b = DisposalTrafficManager(1, slot_steps=20)
    ra = a.reserve(100); b.merge([ra], 0); rb = b.reserve(100)
    assert rb.start_step >= ra.end_step

