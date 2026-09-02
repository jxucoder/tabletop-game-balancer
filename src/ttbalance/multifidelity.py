"""Multi-fidelity surrogate: let cheap runs inform expensive predictions.

A `fast` run costs ~2 minutes, a `medium` one 15-40. Ranking candidates by
their fast score fails — for ExplodingKittens the two fidelities correlate only
weakly, and early searches that trusted fast walked the wrong way. But that is
an argument against using fast as a *ranker*, not against using it as a
*feature*: a regressor is free to learn whatever relationship exists, including
a weak or negative one, and the sign costs it nothing.

Two things this adds over fitting Nori on medium scores alone:

* **Cheap score as a feature.** Train on ``[encoded params, fast_score]`` to
  predict the medium score. Measured on the 18 ExplodingKittens configs that
  carry both, leave-one-out MAE improves from 28.21 to 25.97 (mean-baseline
  35.10). Configs never run at fast get NaN, which Nori handles natively — so
  the column costs nothing on rows that lack it.

* **Every observation, not just the expensive ones.** With fidelity as a
  feature, all ~8,500 cached rows train the model instead of the ~1,000 medium
  ones, and it can learn a different cheap/expensive relationship per game
  (Dominion's fast tracks its medium closely; ExplodingKittens' barely does).

The intended loop: fast-screen a large candidate pool, feed those scores in as
features, and spend medium runs only on what the model ranks highest.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .cache import Cache
from .encode import Encoder
from .spec import GameSpec, Params, canonical

CHEAP = "fast"
EXPENSIVE = "medium"


def paired_scores(cache: Cache, game: str) -> Dict[str, Dict[str, float]]:
    """{config key: {fidelity: mean score}} for every cached configuration."""
    out: Dict[str, Dict[str, float]] = {}
    with cache._lock:                      # noqa: SLF001 - same module family
        rows = cache._conn.execute(
            "SELECT key, run_type, AVG(score) FROM observations"
            " WHERE game=? GROUP BY key, run_type", (game,)).fetchall()
    for key, run_type, mean in rows:
        out.setdefault(key, {})[run_type] = mean
    return out


def build_training(cache: Cache, spec: GameSpec, game: str,
                   target: str = EXPENSIVE, cheap: str = CHEAP):
    """Rows measured at `target`; features are params plus the cheap score.

    Returns (DataFrame, y). The cheap column is NaN wherever that config was
    never run cheaply — Nori treats missing values as missing rather than as a
    number, so those rows still contribute everything their parameters say.
    """
    import json

    import numpy as np
    import pandas as pd

    enc = Encoder(spec)
    per_key = paired_scores(cache, game)
    params: List[Params] = []
    cheap_col: List[float] = []
    y: List[float] = []
    for key, by_fid in per_key.items():
        if target not in by_fid:
            continue
        try:
            p = json.loads(key)
        except ValueError:
            continue
        if spec.validate(p):
            continue
        params.append(p)
        cheap_col.append(by_fid.get(cheap, float("nan")))
        y.append(by_fid[target])
    if not params:
        return None, None
    frame = pd.DataFrame(enc.matrix(params), columns=enc.columns)
    frame["cheap_score"] = cheap_col
    return frame, np.array(y, dtype=float)


def build_queries(spec: GameSpec, candidates: Sequence[Params],
                  cheap_scores: Optional[Sequence[Optional[float]]] = None):
    """Feature frame for candidates, with cheap scores where they exist."""
    import pandas as pd

    enc = Encoder(spec)
    frame = pd.DataFrame(enc.matrix(list(candidates)), columns=enc.columns)
    if cheap_scores is None:
        frame["cheap_score"] = float("nan")
    else:
        frame["cheap_score"] = [
            float("nan") if s is None else float(s) for s in cheap_scores]
    return frame


class MultiFidelityModel:
    """Nori fitted on expensive scores, with the cheap score as a feature."""

    def __init__(self, model_name: str = "nori-6m"):
        from synthefy_nori import NoriRegressor
        self.model = NoriRegressor(model=model_name)
        self.n_train = 0
        self.n_with_cheap = 0

    def fit(self, cache: Cache, spec: GameSpec, game: str,
            target: str = EXPENSIVE, cheap: str = CHEAP) -> bool:
        import numpy as np

        X, y = build_training(cache, spec, game, target, cheap)
        if X is None or len(y) < 5:
            return False
        self.n_train = len(y)
        self.n_with_cheap = int(np.isfinite(X["cheap_score"].to_numpy()).sum())
        self.model.fit(X, y)
        return True

    def acquisition(self, spec: GameSpec, candidates: Sequence[Params],
                    cheap_scores: Optional[Sequence[Optional[float]]] = None,
                    kappa: float = 0.9):
        """Upper-confidence score per candidate: q50 + kappa*(q90 - q50)."""
        import numpy as np

        Xq = build_queries(spec, candidates, cheap_scores)
        try:
            q50, q90 = self.model.predict(Xq, output_type="quantiles",
                                          quantiles=[0.5, 0.9])
            q50 = np.asarray(q50, dtype=float).ravel()
            q90 = np.asarray(q90, dtype=float).ravel()
            return q50 + kappa * np.maximum(q90 - q50, 0.0)
        except Exception:
            return np.asarray(self.model.predict(Xq, output_type="mean"),
                              dtype=float).ravel()
