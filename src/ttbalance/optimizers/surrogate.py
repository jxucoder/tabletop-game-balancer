"""Surrogate-model optimisation with Synthefy's Nori tabular foundation model.

Fast runs cost ~minutes each, so the scarce resource is real evaluations. This
optimiser spends them where a model thinks they will pay off:

1. Fit Nori (in-context regression, no training step) on every configuration we
   have already measured: X = encoded params, y = mean fast score.
2. Propose a large pool of unseen candidates (random + mutations of the current
   leaders) and have Nori predict a score distribution for each in one batch.
   The mutation rate is a trust region: it grows after success streaks and
   shrinks after failure streaks (TuRBO-lite, Eriksson et al. 2019).
3. Pick a batch greedily by an upper-confidence acquisition
   q50 + kappa*(q90 - q50) — exploit high predicted score, explore where the
   model is uncertain — halving the acquisition of near-duplicates of already
   picked rows (local penalisation, Gonzalez et al. 2016) so a batch never
   spends two 40-minute evaluations on the same point.
4. Evaluate only those on the real API — plus a couple of re-observations of
   thinly-measured incumbents for confirmation depth — fold the results back
   in, repeat.

Nori's quantile output is what makes the acquisition possible: it gives a
calibrated notion of "uncertain here" that a plain point regressor cannot.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from ..evaluate import REJECTED, BudgetExhausted, Evaluator
from ..spec import GameSpec, Params, canonical
from .base import OptResult, Tracker


def _observed(ev: Evaluator, spec: GameSpec, min_obs: int = 1):
    """Unique measured configs with their mean score and observation count."""
    import json
    rows = ev.cache.best(ev.game, ev.run_type, limit=100000, min_obs=min_obs)
    out = []
    for key, mean, n, _rt in rows:
        try:
            p = json.loads(key)
        except ValueError:
            continue
        if not spec.validate(p):
            out.append({"params": p, "mean": mean, "n": n, "key": key})
    return out


def _candidates(spec: GameSpec, leaders: Sequence[Params], rng: random.Random,
                pool: int, seen: set, rate: float = 0.2) -> List[Params]:
    """Fresh candidate configs: half random, half mutations of the leaders.

    `rate` is the per-gene mutation rate for leader mutations — the caller
    adapts it across generations (trust-region style), so it is a knob here
    rather than a constant."""
    out: List[Params] = []
    guard = 0
    while len(out) < pool and guard < pool * 20:
        guard += 1
        if leaders and rng.random() < 0.6:
            base = leaders[rng.randrange(len(leaders))]
            cand = spec.mutate(base, rng, rate=rate)
        else:
            cand = spec.sample(rng)
        k = canonical(cand)
        if k in seen:
            continue
        seen.add(k)
        out.append(cand)
    return out


def _select_batch(acq, Xc, batch: int, near_frac: float = 0.85,
                  penalty: float = 0.5) -> List[int]:
    """Greedy local penalisation (Gonzalez et al. 2016): near-identical picks
    waste 40-minute evaluations measuring the same point. After each pick, any
    remaining candidate whose encoded row agrees with the picked row on more
    than `near_frac` of columns has its acquisition multiplied by `penalty` —
    once per near-duplicate conflict — before the next argmax. Only picked rows
    are compared against, so the cost is batch * N row comparisons: cheap next
    to one real evaluation. Returns the picked candidate indices in order."""
    import numpy as np  # lazy, matching the rest of this module
    acq_pen = np.array(acq, dtype=float)
    order: List[int] = []
    for _pick in range(min(batch, len(acq_pen))):
        i = int(np.argmax(acq_pen))
        order.append(i)
        near = np.mean(Xc == Xc[i], axis=1) > near_frac
        acq_pen[near] *= penalty
        acq_pen[i] = -np.inf  # never re-pick the same candidate
    return order


def optimize(spec: GameSpec, ev: Evaluator, rng: random.Random,
             generations: int = 15, pool: int = 4000, batch: int = 12,
             init: int = 40, kappa: float = 0.8, model_name: str = "nori-6m",
             pick_repeats: int = 3, reobserve: int = 2,
             seeds: Optional[Sequence[Params]] = None, state_path: Optional[str] = None,
             **_) -> OptResult:
    import numpy as np  # lazy: only the nori optimiser needs the ML stack
    from synthefy_nori import NoriRegressor

    from ..encode import Encoder

    enc = Encoder(spec)
    tr = Tracker(ev, ev.verbose, "nori")

    # Warm start from anything already in the cache, plus provided seeds.
    for s in (seeds or []):
        sc = ev.evaluate(s)
        if sc != REJECTED:
            tr.offer(s, sc)

    obs = _observed(ev, spec)
    # Defined outside the try: the final result reports these even when the
    # budget dies mid-bootstrap, before the generation loop ever runs.
    n_reobserved = 0
    # Trust-region mutation rate (TuRBO-lite, Eriksson et al. 2019):
    # shrinking concentrates candidates near incumbents when progress
    # stalls locally; expanding escapes plateaus. Two consecutive
    # successes grow the rate, two consecutive failures shrink it.
    mut_rate = 0.2
    n_succ = n_fail = 0
    try:
        # Bootstrap so Nori has enough context to be meaningful.
        if len(obs) < init:
            boot = [spec.sample(rng) for _ in range(init - len(obs))]
            for p, s in zip(boot, ev.evaluate_many(boot)):
                if s != REJECTED:
                    tr.offer(p, s)
            ev.check()
            obs = _observed(ev, spec)

        model = NoriRegressor(model=model_name)

        for gen in range(generations):
            obs = _observed(ev, spec)
            if len(obs) < 5:
                break
            X = np.array(enc.matrix([o["params"] for o in obs]), dtype=float)
            y = np.array([o["mean"] for o in obs], dtype=float)
            model.fit(X, y)

            seen = {o["key"] for o in obs}
            leaders = [o["params"] for o in sorted(obs, key=lambda d: -d["mean"])[:30]]
            cands = _candidates(spec, leaders, rng, pool, seen, rate=mut_rate)
            if not cands:
                break
            Xc = np.array(enc.matrix(cands), dtype=float)
            try:
                q50, q90 = model.predict(Xc, output_type="quantiles",
                                         quantiles=[0.5, 0.9])
                q50 = np.asarray(q50); q90 = np.asarray(q90)
                acq = q50 + kappa * np.maximum(q90 - q50, 0.0)
            except Exception:
                acq = np.asarray(model.predict(Xc, output_type="mean"))

            order = _select_batch(acq, Xc, batch)
            picks = [cands[i] for i in order]
            preds = [float(acq[i]) for i in order]
            if ev.verbose:
                print("  gen %2d/%d: fit on %d obs, scored %d cands, "
                      "top pred %.1f, mut rate %.3f, calls %d" %
                      (gen + 1, generations, len(obs), len(cands),
                       max(preds), mut_rate, ev.n_calls))

            # Evaluate each pick with repeats: denoised feedback keeps the loop
            # from chasing a lucky 36-matchup draw and feeds Nori clean labels,
            # so the surrogate sharpens instead of learning the noise.
            prev_best = tr.best_score  # incumbent before this generation lands
            results = ev.evaluate_many(picks, repeats=pick_repeats)
            hit = 0
            for p, s in zip(picks, results):
                if s != REJECTED:
                    hit += 1
                    tr.offer(p, s)
            ev.check()
            best_now = max((s for s in results if s != REJECTED),
                           default=float("-inf"))
            # Trust-region update: success = this generation's best evaluated
            # pick beat the incumbent. Streaks of 2 trigger the adjustment,
            # and the counters reset so each adjustment needs a fresh streak.
            if best_now > prev_best:
                n_succ += 1
                n_fail = 0
            else:
                n_fail += 1
                n_succ = 0
            if n_succ >= 2:
                mut_rate = min(mut_rate * 1.5, 0.5)
                n_succ = n_fail = 0
            elif n_fail >= 2:
                mut_rate = max(mut_rate * 0.6, 0.05)
                n_succ = n_fail = 0
            if ev.verbose:
                print("     -> %d/%d valid, best this gen %.1f, incumbent %.1f"
                      % (hit, len(picks), best_now, tr.best_score))

            # Re-observe thin incumbents: a lucky noisy draw can crown a
            # config on one observation, so give the highest-mean configs
            # with n < 3 one extra confirmation observation per generation
            # rather than only at the final race. This must be a separate
            # evaluate_many asking for more observations than the cache
            # already holds: observe() returns early once `repeats` scores
            # exist, so folding incumbents into the exploration batch above
            # is a silent no-op whenever pick_repeats <= n (the production
            # medium runs use pick_repeats=1). Requesting n+1 guarantees
            # exactly one new observation each; a config that stays thin is
            # re-selected next generation, stepping its depth toward 3.
            if reobserve > 0:
                thin = [o for o in sorted(obs, key=lambda d: -d["mean"])
                        if o["n"] < 3][:reobserve]
                for want in sorted({o["n"] + 1 for o in thin}):
                    grp = [o["params"] for o in thin if o["n"] + 1 == want]
                    for p, s in zip(grp, ev.evaluate_many(grp, repeats=want)):
                        if s != REJECTED:
                            tr.offer(p, s)
                            n_reobserved += 1  # count only real observations
                ev.check()
    except BudgetExhausted:
        pass

    # Confirm the incumbent and the model's current favourites with repeats.
    try:
        obs = _observed(ev, spec)
        top = [o["params"] for o in sorted(obs, key=lambda d: -d["mean"])[:6]]
        finals = ev.race(top, keep=1, rounds=(3, 7))
        if finals and finals[0][1] > REJECTED:
            tr.offer(finals[0][0], finals[0][1])
    except BudgetExhausted:
        pass

    return tr.result(generations=generations, surrogate=model_name,
                     reobserved=n_reobserved, mut_rate=mut_rate)
