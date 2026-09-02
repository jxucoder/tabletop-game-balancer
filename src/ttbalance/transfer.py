"""Estimate how a medium-fidelity gain transfers to the full-fidelity score.

The leaderboard scores at full (3600 matchups); we optimise at medium (360).
Submission history gives one (medium bundle estimate, full official) pair each,
and because some submissions changed only ONE game, those pairs isolate that
game's transfer rate:

    v2 -> v3   EK only   medium +24.2  ->  full +21.0   rate 0.87
    v3 -> v4   EK only   medium +13.4  ->  full +39.0   rate 2.91
    v4 -> v5   all four  medium +74.4  ->  full  +9.3   rate 0.13

The v5 collapse is the important signal: its Dominion config was submitted at
n=1 (971) but confirms to 956, and Wonders7 sat on a racing boundary. Inflated
medium estimates produce inflated apparent gains that do not exist at full, so
the measured rate craters. Single-game changes off a well-observed baseline
transfer near 1.0; bundles built on thin observations do not.

Practical rule this encodes: submit gains that are confirmed at n>=3, and
prefer changing fewer games per submission so each result stays attributable.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple


def transfer_rate(medium_delta: float, full_delta: float) -> float:
    """Fraction of a medium gain that materialised at full."""
    if abs(medium_delta) < 1e-9:
        return float("nan")
    return full_delta / medium_delta


def project(baseline_full: float, per_game_medium_delta: Dict[str, float],
            rates: Sequence[float] = (0.87, 0.5, 0.13)) -> Dict[float, float]:
    """Project a bundle's full score from a scored baseline plus medium gains.

    Returns {rate: projected_full}. Report the range, not a point estimate —
    with four historical labels the rate is genuinely uncertain, and quoting a
    single number is how the earlier calibrator talked itself into predicting
    a full score above the medium estimate.
    """
    total = sum(per_game_medium_delta.values())
    return {r: baseline_full + total * r for r in rates}


def verdict(projections: Dict[float, float], leader: float) -> str:
    lines = []
    for rate in sorted(projections, reverse=True):
        p = projections[rate]
        margin = p - leader
        lines.append("  rate %.2f -> full ~%.0f  (%s leader by %.0f)"
                     % (rate, p, "beats" if margin > 0 else "short of",
                        abs(margin)))
    return "\n".join(lines)
