"""Noise-aware, cached, parallel evaluation of game configurations.

`fast` runs are only 36 matchups, so a single score is a noisy estimate of the
configuration's true quality. Every optimiser here works against `Evaluator`,
which averages repeated observations and never pays twice for the same work.
"""
from __future__ import annotations

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple

from .cache import Cache
from .client import BaseClient, RunRejected
from .spec import Params, canonical

REJECTED = float("-inf")


class BudgetExhausted(RuntimeError):
    pass


class Evaluator:
    def __init__(self, client: BaseClient, cache: Cache, game: str,
                 run_type: str = "fast", repeats: int = 1, workers: int = 4,
                 budget: Optional[int] = None, verbose: bool = True):
        self.client = client
        self.cache = cache
        self.game = game
        self.run_type = run_type
        self.repeats = repeats
        self.workers = workers
        self.budget = budget
        self.verbose = verbose
        self.n_calls = 0
        self.n_cached = 0
        self.exhausted = False
        self._lock = threading.Lock()
        self._rejected = set()

    # -- budget ---------------------------------------------------------
    def _take(self, n: int = 1) -> None:
        with self._lock:
            if self.budget is not None and self.n_calls + n > self.budget:
                self.exhausted = True
                raise BudgetExhausted(
                    "API budget of %d calls exhausted" % self.budget)
            self.n_calls += n

    def check(self) -> None:
        """Raise if the budget ran out. `evaluate_many` absorbs exhaustion so a
        partly-finished batch is still usable; optimisers call this once they
        have consumed the batch, which is what actually stops their loops."""
        if self.exhausted:
            raise BudgetExhausted("API budget of %d calls exhausted" % self.budget)

    @property
    def remaining(self) -> float:
        return math.inf if self.budget is None else self.budget - self.n_calls

    # -- core -----------------------------------------------------------
    def observe(self, params: Params, run_type: Optional[str] = None,
                repeats: Optional[int] = None) -> List[float]:
        """Ensure at least `repeats` observations exist; return all of them."""
        rt = run_type or self.run_type
        want = self.repeats if repeats is None else repeats
        key = (canonical(params), rt)
        if key in self._rejected:
            return []
        have = self.cache.scores(self.game, params, rt)
        self.n_cached += min(len(have), want)
        while len(have) < want:
            self._take()
            try:
                score = self.client.score(self.game, params, rt)
            except RunRejected as exc:
                self._rejected.add(key)
                if self.verbose:
                    print("  ! rejected: %s" % exc, file=sys.stderr)
                return []
            self.cache.add(self.game, params, rt, score)
            have.append(score)
        return have

    def evaluate(self, params: Params, run_type: Optional[str] = None,
                 repeats: Optional[int] = None) -> float:
        scores = self.observe(params, run_type, repeats)
        return REJECTED if not scores else sum(scores) / len(scores)

    def evaluate_many(self, batch: Sequence[Params], run_type: Optional[str] = None,
                      repeats: Optional[int] = None,
                      on_result: Optional[Callable[[Params, float], None]] = None
                      ) -> List[float]:
        """Evaluate a batch in parallel. Budget exhaustion truncates the batch
        rather than losing the work already done."""
        out: List[float] = [REJECTED] * len(batch)

        def job(i_p: Tuple[int, Params]) -> None:
            i, p = i_p
            try:
                out[i] = self.evaluate(p, run_type, repeats)
            except BudgetExhausted:
                out[i] = REJECTED
            except Exception:
                # a transient backend failure on one candidate drops that
                # candidate, never the whole batch (important for cloud runs).
                out[i] = REJECTED
            if on_result:
                on_result(p, out[i])

        if self.workers <= 1:
            for item in enumerate(batch):
                job(item)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(job, list(enumerate(batch))))
        return out

    # -- noise handling -------------------------------------------------
    def race(self, candidates: Sequence[Params], keep: int = 1,
             rounds: Sequence[int] = (1, 3, 7), run_type: Optional[str] = None
             ) -> List[Tuple[Params, float, int]]:
        """Successive halving over repeated evaluations.

        Cheap single looks at everything, then progressively more repeats on the
        survivors. This is what stops the search from crowning a configuration
        that merely got a lucky 36-matchup draw.
        """
        pool = list(candidates)
        for r, reps in enumerate(rounds):
            if not pool:
                break
            means = self.evaluate_many(pool, run_type=run_type, repeats=reps)
            if self.exhausted and all(m == REJECTED for m in means):
                break
            ranked = sorted(zip(pool, means), key=lambda t: -t[1])
            ranked = [(p, m) for p, m in ranked if m != REJECTED]
            if r < len(rounds) - 1:
                width = max(keep, len(ranked) // 2)
                pool = [p for p, _ in ranked[:width]]
                if self.verbose:
                    print("  race round %d (reps=%d): %d -> %d survivors, "
                          "best %.2f" % (r + 1, reps, len(means), len(pool),
                                         ranked[0][1] if ranked else float("nan")))
            else:
                pool = [p for p, _ in ranked]
        final = []
        for p in pool[:keep]:
            obs = self.cache.scores(self.game, p, run_type or self.run_type)
            final.append((p, sum(obs) / len(obs) if obs else REJECTED, len(obs)))
        return final

    def summary(self) -> str:
        return ("%s/%s: %d API calls, %d cache hits"
                % (self.game, self.run_type, self.n_calls, self.n_cached))
