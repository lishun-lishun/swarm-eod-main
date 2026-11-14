"""V6 event-driven ONE-task coordination. No global auction or hidden state.

Peer commitments are only those received over the official radio. The safety
invariant is: never abandon a still-valid task merely for an uncertain ETA
improvement. Carriers do not participate in bomb bidding.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class KnownTask:
    key: object
    xy: tuple[float, float]
    approach_frames: float
    return_frames: float | None
    grasp_frames: float
    failure_risk_frames: float = 0.0
    feasible: bool = True

    def eta(self, unknown_center_penalty=150.0) -> float:
        if not self.feasible or not math.isfinite(self.approach_frames) or self.approach_frames < 0:
            return math.inf
        home=(self.return_frames if self.return_frames is not None
              and math.isfinite(self.return_frames) else unknown_center_penalty)
        return (self.approach_frames+self.grasp_frames
                +max(0.,home)+max(0.,self.failure_risk_frames))


@dataclass(frozen=True)
class PeerCommitment:
    drone_id: int
    xy: tuple[float,float]
    bid_frames: float | None
    observed_step: int
    carrying: bool = False
    ttl_frames: int = 25

    def fresh(self,now):
        return (0<=now-self.observed_step<=self.ttl_frames)


@dataclass(frozen=True)
class TaskChoice:
    task: KnownTask|None
    reason: str
    total_frames: float


class EventTaskPolicy:
    """Conservative CBBA-inspired *one-task* claim arbitration.

    This is not CBBA convergence: disconnected robots cannot know peer bids.
    Use the official message lease and one-task incumbent throughout loss of
    communication; matching after reconnection resolves duplicate claims.
    """
    def __init__(self,own_id:int, *,switch_gain_abs=80.,switch_gain_fraction=.25,
                 peer_match_px=24.,peer_margin_frames=20.):
        self.own_id=int(own_id)
        self.switch_gain_abs=float(switch_gain_abs)
        self.switch_gain_fraction=float(switch_gain_fraction)
        self.peer_match_px=float(peer_match_px)
        self.peer_margin=float(peer_margin_frames)

    def choose(self, tasks, *, current_key=None, currently_carrying=False,
               current_feasible=True, peers=(), now:int, busy_remaining_frames=0.,
               allow_switch=False) -> TaskChoice:
        if currently_carrying:
            return TaskChoice(None,'CARRIER_CANNOT_ACCEPT_NEW_TASK',math.inf)
        opts=[]
        fresh=[p for p in peers if p.drone_id!=self.own_id and p.fresh(now)]
        for t in tasks:
            eta=t.eta()
            if not math.isfinite(eta):
                continue
            # Include actual *current* remaining work, NOT number of historical
            # pickups; a constant historical penalty changes all bids equally.
            finish=eta+max(0.,busy_remaining_frames)
            owners=[p for p in fresh if math.dist(p.xy,t.xy)<=self.peer_match_px]
            # If a fresh peer claims this specific task, do not steal it unless
            # the compatible peer bid is substantially worse, with deterministic tie.
            blocked=False
            for peer in owners:
                if peer.carrying:
                    blocked=True;break
                if peer.bid_frames is None or not math.isfinite(peer.bid_frames):
                    # Old V3 pixel claim: cannot compare units; respect lease.
                    blocked=True;break
                if (peer.bid_frames<finish+self.peer_margin or
                    (abs(peer.bid_frames-finish)<=self.peer_margin and peer.drone_id<self.own_id)):
                    blocked=True;break
            # Even the incumbent must yield once a verifiably better fresh
            # peer commitment is received; prevent persistent co-claims.
            if not blocked:
                opts.append((finish,repr(t.key),t))
        incumbent=next((o for o in opts if o[2].key==current_key),None)
        if incumbent and current_feasible and not allow_switch:
            return TaskChoice(incumbent[2],'LOCK_VALID_INCUMBENT',incumbent[0])
        if not opts:
            return TaskChoice(None,'NO_LEGAL_UNCLAIMED_TASK',math.inf)
        opts.sort(key=lambda r:(r[0],r[1]))
        best=opts[0]
        if incumbent and current_feasible and best[2].key!=current_key:
            gain=incumbent[0]-best[0]
            need=max(self.switch_gain_abs,self.switch_gain_fraction*incumbent[0])
            if gain<need:
                return TaskChoice(incumbent[2],'SWITCH_REJECTED',incumbent[0])
        return TaskChoice(best[2],'TAKE_UNCLAIMED' if current_key is None else 'REASSIGN_AFTER_EVENT',best[0])
