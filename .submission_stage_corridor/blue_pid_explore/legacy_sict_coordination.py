"""Classical, communication-light coordination helpers for v2.

The controller exchanges only the compact pose/goal/claim summaries already
accepted by the official radio API.  These helpers cluster locally observed
frontiers and apply a deterministic auction utility; they never read map
truth, bomb coordinates, or simulator state.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def cluster_frontiers(points: Iterable[Sequence[float]], radius: float = 80.0,
                      max_clusters: int = 16) -> list[tuple[float, float, int]]:
    """Greedily cluster frontier points into stable centroids.

    The order is sorted lexicographically, making assignments repeatable across
    drones and runs.  Each return value is ``(x, y, population)``.
    """
    pts = sorted((float(p[0]), float(p[1])) for p in points)
    if not pts:
        return []
    r2 = float(radius) ** 2
    clusters: list[list[float]] = []
    counts: list[int] = []
    for x, y in pts:
        best = None
        best_d = r2
        for i, c in enumerate(clusters):
            d = (x - c[0]) ** 2 + (y - c[1]) ** 2
            if d <= best_d:
                best, best_d = i, d
        if best is None:
            if len(clusters) >= max_clusters:
                best = min(range(len(clusters)), key=lambda i: (clusters[i][0] - x) ** 2 + (clusters[i][1] - y) ** 2)
            else:
                clusters.append([x, y]); counts.append(0); best = len(clusters) - 1
        n = counts[best]
        clusters[best][0] = (clusters[best][0] * n + x) / (n + 1)
        clusters[best][1] = (clusters[best][1] * n + y) / (n + 1)
        counts[best] = n + 1
    return [(c[0], c[1], counts[i]) for i, c in enumerate(clusters)]


def auction_utility(drone_id: int, target: Sequence[float], pose: Sequence[float],
                    gain: float, peers: Mapping[int, Mapping], *, switch_cost: float = 30.0) -> float:
    """Return an auction utility (higher is better) for one target."""
    distance = math.dist((float(pose[0]), float(pose[1])), (float(target[0]), float(target[1])))
    utility = float(gain) - 1.0 * distance
    for pid, peer in peers.items():
        peer_goal = peer.get("goal")
        if peer_goal is None or math.dist(peer_goal, target) > 90.0:
            continue
        # Lower identifier wins exact ties, while a carrying drone keeps
        # priority over an explorer at the same frontier.
        if int(pid) < int(drone_id) or peer.get("carry"):
            utility -= switch_cost
    return utility


def repeated_ratio(history: Sequence[Sequence[float]]) -> float:
    """Fraction of selections that revisit an earlier target."""
    seen: set[tuple[float, float]] = set()
    repeated = 0
    for p in history:
        key = (round(float(p[0]), 1), round(float(p[1]), 1))
        if key in seen:
            repeated += 1
        seen.add(key)
    return repeated / max(1, len(history))


