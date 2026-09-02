from . import evolution, hill_climb, pbil, random_search
try:
    from . import surrogate  # optional: needs synthefy-nori + numpy
    _HAS_SURROGATE = True
except Exception:
    _HAS_SURROGATE = False
from .base import OptResult, Tracker

REGISTRY = {
    "random": random_search.optimize,
    "hill": hill_climb.optimize,
    "ea": evolution.optimize,
    "pbil": pbil.optimize,
}
if _HAS_SURROGATE:
    REGISTRY["nori"] = surrogate.optimize

__all__ = ["REGISTRY", "OptResult", "Tracker"]
