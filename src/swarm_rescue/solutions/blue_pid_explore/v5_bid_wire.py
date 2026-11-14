"""Strict, versioned V5 bid transport on top of legal v3 radio messages."""
from __future__ import annotations
from math import isfinite
from typing import Mapping
from .v5_deadline_allocator import TaskBidV5


def encode_bid(bid: TaskBidV5) -> dict:
    return {'protocol': 5, 'drone_id': int(bid.drone_id),
            'position': [float(bid.position[0]), float(bid.position[1])],
            'cost_frames': float(bid.cost_frames),
            'sent_step': int(bid.sent_step), 'ttl_frames': int(bid.ttl_frames)}


def decode_bid(raw, *, sender_id: int, now: int) -> TaskBidV5 | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        if int(raw.get('protocol', -1)) != 5:
            return None
        ident = int(raw.get('drone_id', -1))
        if ident != int(sender_id):
            return None
        pos = raw['position']
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return None
        p = (float(pos[0]), float(pos[1]))
        cost = float(raw['cost_frames'])
        sent = int(raw['sent_step'])
        ttl = int(raw.get('ttl_frames', 25))
        if not all(map(isfinite, (*p, cost))):
            return None
        if ttl <= 0 or ttl > 120 or cost < 0:
            return None
        bid = TaskBidV5(ident, p, cost, sent, ttl)
        return bid if bid.valid(now) else None
    except (TypeError, ValueError, KeyError, OverflowError):
        return None
