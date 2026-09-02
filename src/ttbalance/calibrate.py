"""Calibrate medium-estimate -> full-official score from submission history.

The leaderboard scores at full (3600 matchups); we optimise at medium (360).
Every submission gives one (medium_estimate, full_actual) pair, so a handful of
submissions is enough to fit the medium->full discount and predict what a new
bundle will actually score at full — the number that decides the ranking.

This is deliberately a tiny linear model, not Nori: with ~4 full labels a
foundation model would badly overfit. Nori's place is the per-config surrogate
(hundreds of medium labels); here 4 points want 1-2 parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Submission:
    name: str
    medium_estimate: float
    full_actual: float | None  # None while the server is still evaluating


def _fit_linear(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least-squares y = a + b x. Falls back to a constant offset if the x's
    barely vary (which they do early on)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-6 or n < 3:
        return my - mx, 1.0  # constant discount, slope pinned to 1
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return a, b


def calibrate(subs: Sequence[Submission]) -> dict:
    scored = [s for s in subs if s.full_actual is not None]
    if not scored:
        return {"model": None, "note": "no full labels yet"}
    xs = [s.medium_estimate for s in scored]
    ys = [s.full_actual for s in scored]
    discounts = [s.medium_estimate - s.full_actual for s in scored]
    a, b = _fit_linear(xs, ys)

    # Residual spread as a crude +/- band on predictions.
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    spread = (max(resid) - min(resid)) / 2 if len(resid) > 1 else 20.0

    mean_disc = sum(discounts) / len(discounts)
    worst_disc = max(discounts)   # biggest medium->full drop seen
    # Predict by APPLYING the observed discount, not by extrapolating a slope:
    # a 3-4 point line extrapolates wildly past the data. full ~= medium - disc.
    return {
        "model": (a, b),
        "n_labels": len(scored),
        "discount_mean": mean_disc,
        "discount_range": (min(discounts), max(discounts)),
        "band": max(spread, 8.0),
        "predict": lambda med: med - mean_disc,
        "predict_worst": lambda med: med - worst_disc,
    }


def report(subs: Sequence[Submission], pending_medium: float | None = None,
           leader: float | None = None) -> str:
    cal = calibrate(subs)
    lines = ["medium -> full calibration (%d labels)" % cal.get("n_labels", 0)]
    for s in subs:
        if s.full_actual is not None:
            lines.append("  %-22s medium %.0f -> full %.0f  (discount %+.0f)"
                         % (s.name, s.medium_estimate, s.full_actual,
                            s.full_actual - s.medium_estimate))
        else:
            lines.append("  %-22s medium %.0f -> full  PENDING"
                         % (s.name, s.medium_estimate))
    if cal.get("model") and pending_medium is not None:
        pred = cal["predict"](pending_medium)          # mean-discount estimate
        worst = cal["predict_worst"](pending_medium)   # worst-discount (conservative)
        lines.append("")
        lines.append("  discount seen: mean %+.0f, worst %+.0f (range %+.0f..%+.0f)"
                     % (cal["discount_mean"], max(cal["discount_range"]),
                        *cal["discount_range"]))
        lines.append("  PREDICT full(medium=%.0f): likely %.0f, worst-case %.0f"
                     % (pending_medium, pred, worst))
        if leader is not None:
            m1, m2 = pred - leader, worst - leader
            lines.append("  vs leader %.0f: likely %+.0f, worst-case %+.0f  -> %s"
                         % (leader, m1, m2,
                            "WINS even worst-case" if m2 > 0 else
                            ("WINS likely, worst-case short by %.0f" % -m2 if m1 > 0
                             else "SHORT")))
    return "\n".join(lines)
