"""Offline surrogate of the balancing API.

Lets you develop and benchmark optimisers with zero API calls and no Docker.
It mimics the parts that matter: a score near 1000 at some hidden optimum, a
rugged landscape with interactions, and heteroscedastic noise whose magnitude
depends on the run type (36 / 360 / 3600 matchups).

It is NOT a model of the real games. Use it to check that an optimiser
converges and to tune its hyper-parameters, never to pick final entries.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List

from .client import BaseClient
from .spec import Choice, GameSpec, Params, RUN_TYPES, Subset, load_specs


def _h(*parts: str) -> float:
    """Deterministic pseudo-random float in [0,1) from strings."""
    d = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2 ** 64


class MockClient(BaseClient):
    name = "mock"

    def __init__(self, seed: int = 0, specs: Dict[str, GameSpec] = None,
                 noise_scale: float = 40.0):
        self.seed = str(seed)
        self.specs = specs or load_specs()
        self.noise_scale = noise_scale
        self._rng = random.Random(seed)
        self._targets = {g: self._make_target(g, sp)
                         for g, sp in self.specs.items()}

    def _make_target(self, game: str, spec: GameSpec) -> Params:
        target: Params = {}
        for prm in spec.params:
            r = _h(self.seed, game, prm.name)
            if isinstance(prm, Subset):
                k = prm.min_size + int(r * (prm.max_size - prm.min_size + 1))
                k = min(k, len(prm.options))
                ordered = sorted(prm.options,
                                 key=lambda o: _h(self.seed, game, prm.name, o))
                target[prm.name] = sorted(ordered[:k])
            else:
                target[prm.name] = prm.options[int(r * len(prm.options))]
        return target

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        spec = self.specs[game]
        errs = spec.validate(params)
        if errs:
            from .client import RunRejected
            raise RunRejected("; ".join(errs[:3]))
        target = self._targets[game]

        dist = 0.0
        vals: List[float] = []
        for prm in spec.params:
            v = params[prm.name]
            t = target[prm.name]
            if isinstance(prm, Subset):
                inter = len(set(v) & set(t))
                d = 1.0 - inter / float(max(len(t), 1))
            else:
                i, j = prm.options.index(v), prm.options.index(t)
                d = abs(i - j) / float(max(len(prm.options) - 1, 1))
            vals.append(d)
            dist += d * d
        dist /= len(spec.params)

        # Pairwise interaction so the landscape is not separable.
        inter = 0.0
        for a in range(0, len(vals) - 1, 2):
            inter += vals[a] * vals[a + 1]
        inter /= max(len(vals) / 2.0, 1.0)

        true = 1000.0 * math.exp(-2.4 * dist - 0.6 * inter)
        matchups = RUN_TYPES.get(run_type, 36)
        sigma = self.noise_scale / math.sqrt(matchups / 36.0)
        return round(max(0.0, min(1000.0, true + self._rng.gauss(0, sigma))), 2)

    def optimum(self, game: str) -> Params:
        return dict(self._targets[game])
