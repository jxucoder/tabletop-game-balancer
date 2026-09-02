from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..spec import Params


@dataclass
class OptResult:
    best_params: Params
    best_score: float
    n_calls: int
    elapsed: float
    history: List[Tuple[int, float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Tracker:
    """Records the best-so-far trace and prints progress."""

    def __init__(self, evaluator, verbose: bool = True, label: str = ""):
        self.ev = evaluator
        self.verbose = verbose
        self.label = label
        self.best_params: Optional[Params] = None
        self.best_score = float("-inf")
        self.history: List[Tuple[int, float]] = []
        self.t0 = time.time()

    def offer(self, params: Params, score: float) -> bool:
        if score <= self.best_score:
            return False
        self.best_score = score
        self.best_params = dict(params)
        self.history.append((self.ev.n_calls, score))
        if self.verbose:
            print("  [%5d calls | %6.1fs] new best %.2f%s"
                  % (self.ev.n_calls, time.time() - self.t0, score,
                     (" (%s)" % self.label) if self.label else ""))
        return True

    def result(self, **meta) -> OptResult:
        return OptResult(self.best_params or {}, self.best_score, self.ev.n_calls,
                         time.time() - self.t0, self.history, meta)
