"""Versioned compact team messages for intermittent official radio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TeamMessage:
    sender: int
    step: int
    sequence: int
    pose: tuple[float, float, float]
    velocity: tuple[float, float]
    mission: str
    goal: tuple[float, float] | None
    bomb_updates: tuple[dict[str, Any], ...]
    claims: tuple[dict[str, Any], ...]
    disposal_reservation: dict[str, Any] | None = None
    center: tuple[float, float] | None = None


class CommunicationMemory:
    def __init__(self, sender: int):
        self.sender = int(sender); self.sequence = 0
        self.last_sequence: dict[int, int] = {}; self.peers: dict[int, TeamMessage] = {}

    def make(self, **kwargs) -> dict[str, Any]:
        self.sequence += 1
        message = TeamMessage(sender=self.sender, sequence=self.sequence, **kwargs)
        return message.__dict__.copy()

    def receive(self, payload: Any) -> TeamMessage | None:
        if not isinstance(payload, dict):
            return None
        try:
            message = TeamMessage(**payload)
        except (TypeError, ValueError):
            return None
        if message.sequence <= self.last_sequence.get(message.sender, -1):
            return None
        self.last_sequence[message.sender] = message.sequence; self.peers[message.sender] = message
        return message


