"""(mu + lambda) evolutionary algorithm with noise-aware elitism.

Survivors are re-observed each generation, so a configuration only stays at the
top of the population if it keeps performing. That converts generation count
into evidence, which matters because `fast` runs are 36 matchups of noise.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from ..evaluate import REJECTED, BudgetExhausted, Evaluator
from ..spec import GameSpec, Params
from .base import OptResult, Tracker


def optimize(spec: GameSpec, ev: Evaluator, rng: random.Random,
             generations: int = 40, mu: int = 12, lam: int = 24,
             tournament: int = 3, mutation_rate: float = 0.15,
             crossover_rate: float = 0.6, elite_repeats_cap: int = 9,
             seeds: Optional[Sequence[Params]] = None, **_) -> OptResult:
    tr = Tracker(ev, ev.verbose, "ea")
    pop: List[Tuple[Params, float]] = []
    try:
        init = [dict(s) for s in (seeds or [])][:mu]
        init += [spec.sample(rng) for _ in range(mu - len(init))]
        for p, s in zip(init, ev.evaluate_many(init)):
            if s != REJECTED:
                pop.append((p, s))
                tr.offer(p, s)

        for gen in range(generations):
            if not pop:
                pop = [(p, s) for p, s in
                       zip([spec.sample(rng) for _ in range(mu)],
                           ev.evaluate_many([spec.sample(rng) for _ in range(mu)]))
                       if s != REJECTED]
                if not pop:
                    break

            children: List[Params] = []
            while len(children) < lam:
                a = _tournament(pop, tournament, rng)
                if rng.random() < crossover_rate and len(pop) > 1:
                    b = _tournament(pop, tournament, rng)
                    child = spec.crossover(a, b, rng)
                    child = spec.mutate(child, rng, mutation_rate * 0.5)
                else:
                    child = spec.mutate(a, rng, mutation_rate)
                children.append(child)

            scored = [(p, s) for p, s in zip(children, ev.evaluate_many(children))
                      if s != REJECTED]
            ev.check()

            # Re-observe survivors: more generations survived => more evidence.
            reps = min(elite_repeats_cap, 1 + gen // 4)
            if reps > ev.repeats:
                refreshed = ev.evaluate_many([p for p, _ in pop], repeats=reps)
                pop = [(p, s) for (p, _), s in zip(pop, refreshed)
                       if s != REJECTED]

            merged = sorted(pop + scored, key=lambda t: -t[1])
            pop = _dedupe(merged)[:mu]
            for p, s in pop[:1]:
                tr.offer(p, s)
            if ev.verbose:
                print("  gen %2d/%d: pop best %.2f, median %.2f, calls %d"
                      % (gen + 1, generations, pop[0][1],
                         pop[len(pop) // 2][1], ev.n_calls))
    except BudgetExhausted:
        pass

    # Final confirmation round on the surviving population.
    if pop:
        try:
            finals = ev.race([p for p, _ in pop], keep=1, rounds=(3, 7, 15))
            if finals and finals[0][1] > REJECTED:
                tr.offer(finals[0][0], finals[0][1])
        except BudgetExhausted:
            pass
    return tr.result(mu=mu, lam=lam, generations=generations)


def _tournament(pop: Sequence[Tuple[Params, float]], k: int,
                rng: random.Random) -> Params:
    pick = max(rng.sample(list(pop), min(k, len(pop))), key=lambda t: t[1])
    return pick[0]


def _dedupe(scored: Sequence[Tuple[Params, float]]) -> List[Tuple[Params, float]]:
    from ..spec import canonical
    seen = set()
    out = []
    for p, s in scored:
        k = canonical(p)
        if k not in seen:
            seen.add(k)
            out.append((p, s))
    return out
