"""Parameter space definitions for the four competition games."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

GAMES = ("Dominion", "ExplodingKittens", "Wonders7", "CantStop")
RUN_TYPES = {"fast": 36, "medium": 360, "full": 3600}

_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "valid_params.json",
)

# Params that select a *subset* of the option list rather than a single value.
# (game, param) -> (min_size, max_size)
SUBSET_PARAMS: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("Dominion", "CARDS"): (10, 10),
    ("Wonders7", "wonders"): (4, 7),
}

Params = Dict[str, Any]


@dataclass(frozen=True)
class Choice:
    """A single-value parameter chosen from an ordered option list."""

    name: str
    options: Tuple[Any, ...]

    @property
    def size(self) -> int:
        return len(self.options)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)

    def neighbours(self, value: Any) -> List[Any]:
        """Adjacent options — exploits the fact that option lists are ordered."""
        i = self.options.index(value)
        out = []
        if i > 0:
            out.append(self.options[i - 1])
        if i < len(self.options) - 1:
            out.append(self.options[i + 1])
        return out


@dataclass(frozen=True)
class Subset:
    """A parameter that is a list of distinct names drawn from a pool."""

    name: str
    options: Tuple[str, ...]
    min_size: int
    max_size: int

    def sample(self, rng: random.Random) -> List[str]:
        k = rng.randint(self.min_size, min(self.max_size, len(self.options)))
        return sorted(rng.sample(list(self.options), k))

    def neighbours(self, value: Sequence[str]) -> List[List[str]]:
        """Swap one member out, plus grow/shrink by one where allowed."""
        cur = list(value)
        pool = [o for o in self.options if o not in cur]
        out: List[List[str]] = []
        for i in range(len(cur)):
            for repl in pool:
                cand = sorted(cur[:i] + cur[i + 1:] + [repl])
                out.append(cand)
        if len(cur) < self.max_size:
            for add in pool:
                out.append(sorted(cur + [add]))
        if len(cur) > self.min_size:
            for i in range(len(cur)):
                out.append(sorted(cur[:i] + cur[i + 1:]))
        return out


Param = Any  # Choice | Subset


class GameSpec:
    """The searchable parameter space for one game."""

    def __init__(self, game: str, raw: Dict[str, List[Any]]):
        self.game = game
        self.params: List[Param] = []
        for name, options in raw.items():
            key = (game, name)
            if key in SUBSET_PARAMS:
                lo, hi = SUBSET_PARAMS[key]
                self.params.append(Subset(name, tuple(options), lo, hi))
            else:
                self.params.append(Choice(name, tuple(options)))
        self.by_name = {p.name: p for p in self.params}

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def sample(self, rng: random.Random) -> Params:
        return {p.name: p.sample(rng) for p in self.params}

    def default(self) -> Params:
        """Midpoint of every option list — a neutral starting configuration."""
        out: Params = {}
        for p in self.params:
            if isinstance(p, Subset):
                k = min(p.max_size, len(p.options))
                out[p.name] = sorted(list(p.options)[:k])
            else:
                out[p.name] = p.options[len(p.options) // 2]
        return out

    def neighbours(self, params: Params) -> Iterator[Params]:
        """All single-parameter mutations of `params`."""
        for p in self.params:
            for alt in p.neighbours(params[p.name]):
                cand = dict(params)
                cand[p.name] = alt
                yield cand

    def mutate(self, params: Params, rng: random.Random, rate: float = 0.15,
               local: float = 0.7) -> Params:
        """Per-gene mutation. `local` = probability a Choice steps to an
        adjacent option instead of jumping anywhere in its range."""
        out = dict(params)
        changed = False
        for p in self.params:
            if rng.random() >= rate:
                continue
            changed = True
            if isinstance(p, Subset):
                out[p.name] = rng.choice(p.neighbours(out[p.name]))
            elif rng.random() < local:
                nb = p.neighbours(out[p.name])
                out[p.name] = rng.choice(nb) if nb else out[p.name]
            else:
                out[p.name] = p.sample(rng)
        if not changed:  # never return an identical child
            order = list(self.params)
            rng.shuffle(order)
            for p in order:
                cur = out[p.name]
                if isinstance(p, Subset):
                    alts = [a for a in p.neighbours(cur) if sorted(a) != sorted(cur)]
                else:
                    # Resampling can land on the current value, so exclude it.
                    alts = [o for o in p.options if o != cur]
                if alts:
                    out[p.name] = rng.choice(alts)
                    break
        return out

    def crossover(self, a: Params, b: Params, rng: random.Random) -> Params:
        return {p.name: (a if rng.random() < 0.5 else b)[p.name] for p in self.params}

    def validate(self, params: Params) -> List[str]:
        errs = []
        for name, value in params.items():
            p = self.by_name.get(name)
            if p is None:
                errs.append("unknown param %r" % name)
                continue
            if isinstance(p, Subset):
                if not isinstance(value, list):
                    errs.append("%s must be a list" % name)
                elif len(set(value)) != len(value):
                    errs.append("%s has duplicates" % name)
                elif not (p.min_size <= len(value) <= p.max_size):
                    errs.append("%s size %d outside [%d,%d]"
                                % (name, len(value), p.min_size, p.max_size))
                elif any(v not in p.options for v in value):
                    errs.append("%s has values outside the pool" % name)
            elif value not in p.options:
                errs.append("%s=%r not in %r" % (name, value, list(p.options)))
        for p in self.params:
            if p.name not in params:
                errs.append("missing param %s" % p.name)
        return errs

    def size(self) -> float:
        n = 1.0
        for p in self.params:
            n *= len(p.options) if isinstance(p, Choice) else 1e5
        return n


def load_specs(path: str = _CONFIG) -> Dict[str, GameSpec]:
    with open(path) as fh:
        raw = json.load(fh)
    return {g: GameSpec(g, raw[g]) for g in raw}


def canonical(params: Params) -> str:
    """Stable key for caching — lists are sorted so member order never matters."""
    norm = {k: (sorted(v) if isinstance(v, list) else v) for k, v in params.items()}
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))
