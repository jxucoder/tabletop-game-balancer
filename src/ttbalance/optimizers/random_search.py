"""Uniform random search — the baseline every other optimiser must beat."""
from __future__ import annotations

import random
from typing import Optional

from ..evaluate import BudgetExhausted, Evaluator
from ..spec import GameSpec, Params
from .base import OptResult, Tracker


def optimize(spec: GameSpec, ev: Evaluator, rng: random.Random,
             iterations: int = 200, batch: int = 8,
             seed_params: Optional[Params] = None, **_) -> OptResult:
    tr = Tracker(ev, ev.verbose, "random")
    if seed_params:
        tr.offer(seed_params, ev.evaluate(seed_params))
    done = 0
    try:
        while done < iterations:
            n = min(batch, iterations - done)
            cands = [spec.sample(rng) for _ in range(n)]
            for p, s in zip(cands, ev.evaluate_many(cands)):
                tr.offer(p, s)
            done += n
            ev.check()
    except BudgetExhausted:
        pass
    return tr.result(iterations=done)
