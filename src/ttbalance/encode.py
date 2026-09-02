"""Encode game configurations as a fixed-width numeric table for a surrogate.

The parameter space mixes ordered integer choices, one boolean, and two
set-valued params (Dominion CARDS = 10 of 26, Wonders7 wonders = 4-7 of 7).
A tabular regressor needs a fixed-width row, so:

* Choice params  -> one numeric column holding the chosen value (ordered, so
  numeric distance is meaningful).
* Subset params  -> one 0/1 column per pool option (multi-hot presence).

The same GameSpec drives encode and the reverse, so columns line up across the
observed table and the candidate table Nori scores against.
"""
from __future__ import annotations

from typing import Dict, List

from .spec import Choice, GameSpec, Params, Subset


class Encoder:
    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.columns: List[str] = []
        self._kind: Dict[str, tuple] = {}  # col -> ("choice", name) | ("subset", name, opt)
        for p in spec.params:
            if isinstance(p, Subset):
                for opt in p.options:
                    col = "%s::%s" % (p.name, opt)
                    self.columns.append(col)
                    self._kind[col] = ("subset", p.name, opt)
            else:
                self.columns.append(p.name)
                self._kind[p.name] = ("choice", p.name)

    def row(self, params: Params) -> List[float]:
        out: List[float] = []
        for col in self.columns:
            kind = self._kind[col]
            if kind[0] == "choice":
                v = params[kind[1]]
                out.append(float(v) if not isinstance(v, bool) else float(bool(v)))
            else:
                _, name, opt = kind
                out.append(1.0 if opt in params.get(name, []) else 0.0)
        return out

    def matrix(self, configs: List[Params]) -> List[List[float]]:
        return [self.row(c) for c in configs]

    @property
    def categorical_columns(self) -> List[str]:
        """Choice columns are ordered ints; multi-hot subset columns are 0/1.
        None are free-categorical, so Nori can treat everything numeric."""
        return []
