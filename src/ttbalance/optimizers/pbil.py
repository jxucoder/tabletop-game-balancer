"""Population-Based Incremental Learning over the categorical parameter space.

Instead of carrying individual configurations forward, PBIL carries a
probability distribution per parameter and nudges it toward whatever the best
samples of each generation chose. Averaging over a whole generation makes it
markedly more robust to evaluation noise than point-based local search, and the
learned marginals double as a readable report of which settings actually matter.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence

from ..evaluate import REJECTED, BudgetExhausted, Evaluator
from ..spec import Choice, GameSpec, Params, Subset
from .base import OptResult, Tracker


class Marginals:
    """One probability vector per parameter over its option list."""

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.p: Dict[str, List[float]] = {
            prm.name: [1.0 / len(prm.options)] * len(prm.options)
            for prm in spec.params
        }

    def sample(self, rng: random.Random) -> Params:
        out: Params = {}
        for prm in self.spec.params:
            probs = self.p[prm.name]
            if isinstance(prm, Subset):
                k = rng.randint(prm.min_size, min(prm.max_size, len(prm.options)))
                out[prm.name] = sorted(_weighted_sample(prm.options, probs, k, rng))
            else:
                out[prm.name] = _weighted_pick(prm.options, probs, rng)
        return out

    def update(self, winners: Sequence[Params], lr: float, mutate: float,
               rng: random.Random) -> None:
        for prm in self.spec.params:
            probs = self.p[prm.name]
            target = [0.0] * len(probs)
            for w in winners:
                val = w[prm.name]
                chosen = val if isinstance(val, list) else [val]
                for v in chosen:
                    target[prm.options.index(v)] += 1.0
            total = sum(target) or 1.0
            target = [t / total for t in target]
            for i in range(len(probs)):
                probs[i] += lr * (target[i] - probs[i])
                if rng.random() < mutate:
                    probs[i] += rng.uniform(-0.05, 0.05)
            probs[:] = _renorm(probs, floor=0.01)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"game": self.spec.game, "p": self.p}, fh)
        os.replace(tmp, path)          # atomic: a killed pass cannot corrupt it

    def load(self, path: str) -> bool:
        """Restore marginals from a previous pass. Silently ignores a file that
        no longer matches the parameter space."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (ValueError, OSError):
            return False
        if data.get("game") != self.spec.game:
            return False
        for prm in self.spec.params:
            vec = data.get("p", {}).get(prm.name)
            if not isinstance(vec, list) or len(vec) != len(prm.options):
                return False
        self.p = {k: list(v) for k, v in data["p"].items()}
        return True

    def report(self, top: int = 3) -> str:
        lines = []
        for prm in self.spec.params:
            probs = self.p[prm.name]
            order = sorted(range(len(probs)), key=lambda i: -probs[i])[:top]
            lines.append("    %-32s %s" % (prm.name, ", ".join(
                "%s:%.2f" % (prm.options[i], probs[i]) for i in order)))
        return "\n".join(lines)


def optimize(spec: GameSpec, ev: Evaluator, rng: random.Random,
             generations: int = 40, samples: int = 10, elite_frac: float = 0.3,
             lr: float = 0.25, mutate: float = 0.02,
             seeds: Optional[Sequence[Params]] = None,
             state_path: Optional[str] = None, **_) -> OptResult:
    tr = Tracker(ev, ev.verbose, "pbil")
    m = Marginals(spec)
    resumed = m.load(state_path) if state_path else False
    if resumed and ev.verbose:
        print("  resumed learned marginals from %s" % state_path)
    if seeds and not resumed:
        m.update([dict(s) for s in seeds], lr=0.5, mutate=0.0, rng=rng)
    n_elite = max(2, int(samples * elite_frac))
    try:
        for gen in range(generations):
            batch = [m.sample(rng) for _ in range(samples)]
            scores = ev.evaluate_many(batch)
            ev.check()
            ranked = sorted([(p, s) for p, s in zip(batch, scores) if s != REJECTED],
                            key=lambda t: -t[1])
            if not ranked:
                continue
            tr.offer(ranked[0][0], ranked[0][1])
            m.update([p for p, _ in ranked[:n_elite]], lr, mutate, rng)
            if state_path:
                m.save(state_path)     # after every generation, not at the end
            if ev.verbose:
                print("  gen %2d/%d: best %.2f, elite mean %.2f, calls %d"
                      % (gen + 1, generations, ranked[0][1],
                         sum(s for _, s in ranked[:n_elite]) / n_elite, ev.n_calls))
    except BudgetExhausted:
        pass

    # Greedy configuration from the learned marginals, then confirm.
    try:
        greedy: Params = {}
        for prm in spec.params:
            probs = m.p[prm.name]
            order = sorted(range(len(probs)), key=lambda i: -probs[i])
            if isinstance(prm, Subset):
                k = min(prm.max_size, len(prm.options))
                greedy[prm.name] = sorted(prm.options[i] for i in order[:k])
            else:
                greedy[prm.name] = prm.options[order[0]]
        cands = [greedy] + ([tr.best_params] if tr.best_params else [])
        for p, s, _ in ev.race(cands, keep=len(cands), rounds=(3, 9)):
            tr.offer(p, s)
    except BudgetExhausted:
        pass
    res = tr.result(generations=generations, samples=samples)
    res.meta["marginals"] = m.report()
    return res


def _renorm(probs: List[float], floor: float) -> List[float]:
    probs = [max(floor, p) for p in probs]
    total = sum(probs)
    return [p / total for p in probs]


def _weighted_pick(options, probs, rng: random.Random):
    r = rng.random() * sum(probs)
    acc = 0.0
    for opt, p in zip(options, probs):
        acc += p
        if r <= acc:
            return opt
    return options[-1]


def _weighted_sample(options, probs, k: int, rng: random.Random) -> List:
    """Sample k distinct options without replacement, proportional to probs."""
    pool = list(options)
    weights = list(probs)
    out = []
    for _ in range(min(k, len(pool))):
        pick = _weighted_pick(pool, weights, rng)
        i = pool.index(pick)
        out.append(pool.pop(i))
        weights.pop(i)
    return out
