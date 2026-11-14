"""V5 wire compatibility with the existing pixel-priced claim protocol."""

from swarm_rescue.solutions.blue_pid_explore.legacy_sict_knowledge import Knowledge
from swarm_rescue.solutions.blue_pid_explore.v5_deadline_allocator import (
    DeadlineTaskScheduler, TaskCandidate,
)


class _Grid:
    center_points = ()
    seen_points = ()
    center_bad = ()

    @staticmethod
    def all_sources():
        return {}

    @staticmethod
    def add_center_bad(_):
        pass

    @staticmethod
    def add_center_points(_):
        pass

    @staticmethod
    def add_seen_points(_):
        pass

    @staticmethod
    def merge_sources(_):
        pass


def _message(knowledge, step, claim):
    return knowledge.message(step, (0, 0), (0, 0), False, "FETCH",
                             None, claim, _Grid(), 5)


def test_v5_bid_is_separate_from_legacy_pixel_claim():
    sender = Knowledge(1)
    receiver = Knowledge(2)
    task = TaskCandidate((1, 7), (100.0, 0.0), 200.0, 400.0)
    decision = DeadlineTaskScheduler(1).choose(
        [task], now=100, horizon=5400, incumbent_key=None)
    sender.v5_outgoing_bid = DeadlineTaskScheduler(1).make_bid(decision, now=100)
    message = _message(sender, 100, ((100.0, 0.0), 200.0))
    assert message["claim"][1] == 200.0
    assert message["v5_bid"]["cost_frames"] == decision.score_frames
    receiver.receive([(1, message)], 100, _Grid())
    assert receiver.v5_incoming_bids[1].cost_frames == decision.score_frames
    assert receiver.claimed_by_better_peer((100.0, 0.0), 250.0)
    assert not receiver.claimed_by_better_peer(
        (100.0, 0.0), 250.0, ignore_peer_ids={1})
    receiver.receive((), 126, _Grid())
    assert receiver.v5_incoming_bids == {}


def test_unversioned_peer_keeps_legacy_exclusion():
    receiver = Knowledge(2)
    receiver.receive([(1, _message(Knowledge(1), 100,
                                   ((30.0, 0.0), 80.0)))], 100, _Grid())
    assert receiver.v5_incoming_bids == {}
    assert receiver.claimed_by_better_peer((30.0, 0.0), 200.0)
