"""Bomb tracks, tombstones, claims and peer states shared by radio gossip.

Radio range is limited (250 px), so knowledge travels hop by hop: every
message carries the sender's complete bomb list, recent tombstones and every
map snapshot it knows (see ``GridMap.all_sources``).

Bomb identity is purely geometric: two reports closer than ``MATCH`` are the
same bomb. A tombstone (position, step) deletes every track near it that was
last seen *before* the tombstone, so a bomb that is dropped or pushed and later
sighted again (newer sighting) comes back automatically.
"""

from __future__ import annotations

import math
from .v5_bid_wire import encode_bid, decode_bid

MATCH = 24.0          # bomb radius is 12 px; two legal bombs can touch
TOMB_RADIUS = 22.0
CLAIM_TTL = 25        # steps a peer claim stays valid without refresh
PEER_TTL = 12


class Knowledge:
    def __init__(self, own_id):
        self.own_id = own_id
        self.bombs = {}        # key -> {"pos", "seen", "w"}
        self.tombs = []        # [(x, y, step)]
        self.peers = {}        # id -> state dict (direct radio contact)
        self.claims = {}       # id -> (pos, cost, step)
        self.v5_outgoing_bid = None
        self.v5_incoming_bids = {}
        self.failures = {}     # bomb key -> local failed attempts
        self.next_key = 0
        self.disposed = 0
        self.pickups = {}      # "id:step" -> (x, y), gossiped
        self.drops = set()     # pickup keys whose bomb was dropped / handed over

    # ----------------------------------------------------------------- bombs
    def _find(self, pos, radius=MATCH):
        best, best_d = None, radius
        for key, track in self.bombs.items():
            d = math.dist(track["pos"], pos)
            if d < best_d:
                best, best_d = key, d
        return best

    def _tombstoned(self, pos, seen):
        return any(step >= seen and math.dist((x, y), pos) < TOMB_RADIUS
                   for x, y, step in self.tombs)

    @staticmethod
    def _keep_view(track, view):
        """Keep the closest viewpoint (x, y, dist) a bomb was sighted from.

        A drone stood there and had semantic line of sight, so the spot is
        reachable and looks at the bomb even when the coarse map draws the
        slot the bomb sits in as wall.
        """
        if view is None:
            return
        old = track.get("view")
        if old is None or view[2] < old[2]:
            track["view"] = (float(view[0]), float(view[1]), float(view[2]))

    def observe_bomb(self, pos, step, weight, view=None):
        """Own semantic sighting (weight ~ 1/distance quality)."""
        key = self._find(pos)
        if key is None:
            if self._tombstoned(pos, step):
                return None
            key = (self.own_id, self.next_key)
            self.next_key += 1
            self.bombs[key] = {"pos": pos, "seen": step, "w": weight}
            self._keep_view(self.bombs[key], view)
            return key
        track = self.bombs[key]
        self._keep_view(track, view)
        w_old = track["w"] * (0.8 if step > track["seen"] else 1.0)
        total = w_old + weight
        track["pos"] = ((track["pos"][0] * w_old + pos[0] * weight) / total,
                        (track["pos"][1] * w_old + pos[1] * weight) / total)
        track["w"] = min(total, 20.0)
        track["seen"] = step
        return key

    def remote_bomb(self, pos, seen, view=None):
        if self._tombstoned(pos, seen):
            return
        key = self._find(pos)
        if key is None:
            key = (self.own_id, self.next_key)
            self.next_key += 1
            self.bombs[key] = {"pos": pos, "seen": seen, "w": 0.5}
            self._keep_view(self.bombs[key], view)
            return
        track = self.bombs[key]
        self._keep_view(track, view)
        if seen > track["seen"] + 30:
            # Much newer report: the bomb may have moved; trust it.
            track["pos"] = pos
            track["w"] = 0.5
        track["seen"] = max(track["seen"], seen)

    def tombstone(self, pos, step):
        self.tombs.append((pos[0], pos[1], step))
        if len(self.tombs) > 60:
            self.tombs = self.tombs[-60:]
        for key in [k for k, t in self.bombs.items()
                    if math.dist(t["pos"], pos) < TOMB_RADIUS and t["seen"] <= step]:
            del self.bombs[key]

    def remote_tomb(self, x, y, step):
        if any(abs(step - s) < 5 and math.dist((x, y), (tx, ty)) < 6
               for tx, ty, s in self.tombs):
            return
        self.tombstone((x, y), step)

    # ----------------------------------------------------------------- radio
    def message(self, step, pose, vel, carrying, state, goal, claim, grid,
                health, pickup=None):
        msg = {
            "v3": 1, "id": self.own_id, "step": step,
            "pos": pose, "vel": vel, "carry": carrying, "state": state,
            "goal": goal, "claim": claim, "health": health,
            "bombs": [(t["pos"][0], t["pos"][1], t["seen"]) + tuple(t.get("view") or ())
                      for t in self.bombs.values()],
            "tombs": self.tombs[-30:],
            "center": grid.center_points,
            "seen": grid.seen_points,
            "center_bad": list(grid.center_bad),
            "grids": grid.all_sources(),
            "disposed": self.disposed,
            "pickup": pickup,
            "pickups": self.pickups,
            "drops": list(self.drops),
        }
        if self.v5_outgoing_bid is not None and self.v5_outgoing_bid.valid(step):
            msg["v5_bid"] = encode_bid(self.v5_outgoing_bid)
        return msg

    def receive(self, messages, step, grid):
        for _, msg in messages or ():
            if not isinstance(msg, dict) or msg.get("v3") != 1:
                continue
            pid = msg.get("id")
            if pid == self.own_id:
                continue
            parsed_bid = decode_bid(msg.get("v5_bid"), sender_id=pid, now=step)
            if parsed_bid is not None:
                self.v5_incoming_bids[pid] = parsed_bid
            else:
                self.v5_incoming_bids.pop(pid, None)
            self.peers[pid] = {"pos": msg.get("pos"), "vel": msg.get("vel"),
                               "carry": msg.get("carry"),
                               "state": msg.get("state"),
                               "goal": msg.get("goal"), "step": step,
                               "pickup": msg.get("pickup"),
                               "health": msg.get("health")}
            claim = msg.get("claim")
            if claim is not None:
                self.claims[pid] = (claim[0], claim[1], step)
            else:
                self.claims.pop(pid, None)
            for x, y, s in msg.get("tombs", ()):
                self.remote_tomb(x, y, s)
            for item in msg.get("bombs", ()):
                self.remote_bomb((item[0], item[1]), item[2],
                                 tuple(item[3:6]) if len(item) >= 6 else None)
            grid.add_center_bad(msg.get("center_bad", ()))
            grid.add_center_points(msg.get("center", ()))
            grid.add_seen_points(msg.get("seen", ()))
            grid.merge_sources(msg.get("grids", {}))
            self.disposed = max(self.disposed, msg.get("disposed", 0))
            for key, pos in (msg.get("pickups") or {}).items():
                self.pickups.setdefault(key, pos)
            self.drops.update(msg.get("drops") or ())
        self.peers = {k: v for k, v in self.peers.items()
                      if step - v["step"] <= PEER_TTL}
        self.claims = {k: v for k, v in self.claims.items()
                       if step - v[2] <= CLAIM_TTL}
        self.v5_incoming_bids = {
            k: v for k, v in self.v5_incoming_bids.items()
            if k in self.peers and v.valid(step)
        }

    def accounted(self):
        """Bombs picked up (carried or delivered) plus bombs still on the list."""
        picked = sum(1 for key in self.pickups if key not in self.drops)
        return picked + len(self.bombs)

    # ---------------------------------------------------------------- claims
    def claimed_by_better_peer(self, pos, my_cost, ignore_peer_ids=()):
        """A teammate is already going for this bomb and is (nearly) closer."""
        for pid, (cpos, cost, _) in self.claims.items():
            if pid in ignore_peer_ids:
                continue
            if cpos is None or math.dist(cpos, pos) > MATCH + 6:
                continue
            if cost < my_cost - 25 or (abs(cost - my_cost) <= 25 and pid < self.own_id):
                return True
        return False


