"""Restarting steepest-ascent hill climber over the discrete option lattice.

Improvements over the tutorial climber: neighbours are evaluated in parallel,
every incumbent is re-observed so noise cannot lock in a lucky draw, and the
search restarts from a fresh random point once it plateaus.
"""
from __future__ import annotations

import random
from typing import Optional

from ..evaluate import REJECTED, BudgetExhausted, Evaluator
from ..spec import GameSpec, Params
from .base import OptResult, Tracker


def optimize(spec: GameSpec, ev: Evaluator, rng: random.Random,
             restarts: int = 6, max_steps: int = 60, sample_neighbours: int = 0,
             confirm_repeats: int = 3, seed_params: Optional[Params] = None,
             **_) -> OptResult:
    tr = Tracker(ev, ev.verbose, "hill")
    try:
        for r in range(restarts):
            current = (dict(seed_params) if (r == 0 and seed_params)
                       else spec.sample(rng))
            cur_score = ev.evaluate(current)
            if cur_score == REJECTED:
                continue
            tr.offer(current, cur_score)
            for _ in range(max_steps):
                nbrs = list(spec.neighbours(current))
                if sample_neighbours and len(nbrs) > sample_neighbours:
                    nbrs = rng.sample(nbrs, sample_neighbours)
                scores = ev.evaluate_many(nbrs)
                ev.check()
                best_i = max(range(len(scores)), key=lambda i: scores[i]) \
                    if scores else -1
                if best_i < 0 or scores[best_i] <= cur_score:
                    break  # local optimum
                # Confirm the winner with extra observations before committing.
                cand = nbrs[best_i]
                confirmed = ev.evaluate(cand, repeats=confirm_repeats)
                cur_score = ev.evaluate(current, repeats=confirm_repeats)
                if confirmed <= cur_score:
                    break
                current, cur_score = cand, confirmed
                tr.offer(current, cur_score)
            if ev.verbose:
                print("  restart %d/%d done (best %.2f)" % (r + 1, restarts,
                                                            tr.best_score))
    except BudgetExhausted:
        pass
    return tr.result(restarts=restarts)
