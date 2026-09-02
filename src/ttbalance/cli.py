"""Command line entry point: python -m ttbalance <command>"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

from .cache import Cache
from .client import ApiError, RunRejected, make_client
from .evaluate import REJECTED, Evaluator
from .mock import MockClient
from .optimizers import REGISTRY
from .spec import GAMES, RUN_TYPES, GameSpec, Params, canonical, load_specs

DEFAULT_HOSTED = "https://balance-competition.tabletopgames.ai/api/"
DEFAULT_LOCAL = "http://localhost:3000/api/"


# ---------------------------------------------------------------- helpers
def build_client(args):
    backend = args.backend
    if backend == "mock":
        return MockClient(seed=args.seed)
    if backend == "local":
        from .client import parse_pool_spec
        urls = None
        if getattr(args, "pool_url", ""):
            # one autoscaling endpoint (e.g. Modal): N client sessions, one
            # request in flight each, all to the same URL -> the endpoint scales
            # to N workers behind it.
            base = args.pool_url.rstrip("/") + "/"
            urls = [base] * max(1, args.pool_replicas)
        elif getattr(args, "pool_hosts", ""):
            urls = parse_pool_spec(args.pool_hosts)
        return make_client("local", local_url=args.local_url or DEFAULT_LOCAL,
                           pool=args.local_pool, first_port=args.first_port,
                           pool_url_list=urls,
                           timeout_ms=args.timeout_ms or None)
    if backend == "modal":
        return make_client("modal", timeout_ms=args.timeout_ms or None)
    key = args.api_key or os.environ.get("TTB_API_KEY", "")
    if not key:
        sys.exit("error: hosted backend needs an API key. Set TTB_API_KEY or "
                 "pass --api-key. Generate one at\n  "
                 "https://balance-competition.tabletopgames.ai/settings")
    return make_client("hosted", api_key=key,
                       hosted_url=args.hosted_url or DEFAULT_HOSTED)


def games_from(args) -> List[str]:
    if not args.game or args.game == ["all"]:
        return list(GAMES)
    for g in args.game:
        if g not in GAMES:
            sys.exit("error: unknown game %r (choose from %s)" % (g, ", ".join(GAMES)))
    return args.game


def entry_path(game: str) -> str:
    return os.path.join("results", "entries", "%s.json" % game)


FIDELITY = {"fast": 0, "medium": 1, "full": 2}


def save_entry(game: str, params: Params, score: float, run_type: str,
               n_obs: int, backend: str) -> str:
    path = entry_path(game)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = None
    if os.path.exists(path):
        with open(path) as fh:
            prev = json.load(fh)
    record = {"game": game, "params": params, "score": score,
              "run_type": run_type, "observations": n_obs, "backend": backend,
              "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if prev:
        old_f = FIDELITY.get(prev.get("run_type", "fast"), 0)
        new_f = FIDELITY.get(run_type, 0)
        old_n = prev.get("observations", 0)
        old_s = prev.get("score", REJECTED)
        # A higher-fidelity measurement always wins: 360 matchups tell you more
        # than any number of 36-matchup draws.
        if new_f < old_f:
            return path
        if new_f == old_f:
            # At equal fidelity, never let a LESS-observed config replace a
            # better-confirmed one — a lucky n=2 draw reading 961 must not evict
            # a confirmed n=7 at 941. Overwrite only when the newcomer is at
            # least as well observed and genuinely higher.
            if n_obs < old_n:
                return path
            if score <= old_s and n_obs == old_n:
                return path
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    return path


def pick_entry(cache: Cache, spec: GameSpec, game: str, run_type: str,
               fallback: Optional[Params], min_obs: int = 2):
    """Choose the configuration to record.

    Deliberately NOT the single best score ever observed — that is the value
    most inflated by noise. Prefer the best *mean* among configurations that
    have been observed more than once, and report that mean.
    """
    for key, mean, n, _rt in cache.best(game, run_type, limit=5, min_obs=min_obs):
        params = json.loads(key)
        if not spec.validate(params):
            return params, mean, n
    if fallback is None:
        return None, REJECTED, 0
    obs = cache.scores(game, fallback, run_type)
    return fallback, (sum(obs) / len(obs) if obs else REJECTED), len(obs)


# ---------------------------------------------------------------- commands
def cmd_search(args) -> None:
    specs = load_specs()
    cache = Cache(args.cache)
    client = build_client(args)
    rng = random.Random(args.seed)
    opt = REGISTRY[args.optimizer]

    for game in games_from(args):
        spec = specs[game]
        print("\n=== %s | %s | backend=%s run_type=%s ===" %
              (game, args.optimizer, client.name, args.run_type))
        ev = Evaluator(client, cache, game, run_type=args.run_type,
                       repeats=args.repeats, workers=args.workers,
                       budget=args.budget, verbose=not args.quiet)
        seeds = load_seeds(cache, spec, game, args.run_type, args.seed_from_cache)
        kwargs = dict(json.loads(args.opt_args)) if args.opt_args else {}
        if args.optimizer == "pbil":
            # Persist the learned distribution so short repeated passes keep
            # compounding instead of restarting from uniform every time.
            kwargs.setdefault("state_path",
                              os.path.join("results", "state",
                                           "pbil_%s.json" % game))
        if seeds:
            kwargs.setdefault("seeds", seeds)
            kwargs.setdefault("seed_params", seeds[0])
        try:
            res = opt(spec, ev, rng, iterations=args.iterations,
                      generations=args.generations, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Docker down, connection reset, a bug in one optimiser — none of it
            # should abort the other games or crash an unattended overnight run.
            print("  interrupted (%s): %s" % (type(exc).__name__, exc))
            res = None
        print("  %s" % ev.summary())
        if res and res.best_params:
            if args.optimizer == "pbil" and res.meta.get("marginals"):
                print("  learned marginals:\n%s" % res.meta["marginals"])
            params, mean, n_obs = pick_entry(cache, spec, game, args.run_type,
                                             res.best_params)
            print("  best single observation %.2f (noise-inflated)"
                  % res.best_score)
            print("  recorded %.2f as the mean of %d observation%s%s"
                  % (mean, n_obs, "" if n_obs == 1 else "s",
                     "  <- run `verify` before trusting this"
                     if n_obs < 2 else ""))
            if params is not None:
                print("  saved -> %s" % save_entry(game, params, mean,
                                                   args.run_type, n_obs,
                                                   client.name))


def load_seeds(cache: Cache, spec: GameSpec, game: str, run_type: str,
               n: int) -> List[Params]:
    """Warm-start from the best configurations already in the cache."""
    if n <= 0:
        return []
    out: List[Params] = []
    for key, _mean, _n, _rt in cache.best(game, run_type, limit=n):
        try:
            params = json.loads(key)
        except ValueError:
            continue
        if not spec.validate(params):
            out.append(params)
    # The midpoint configuration is a genuinely strong starting point here —
    # CantStop's default measured 895/1000 against 728 for a random draw — so
    # always offer it to the optimiser rather than relying on a lucky sample.
    default = spec.default()
    if canonical(default) not in {canonical(p) for p in out}:
        out.append(default)
    return out


def cmd_verify(args) -> None:
    """Re-evaluate the top cached configurations at a higher fidelity."""
    specs = load_specs()
    cache = Cache(args.cache)
    client = build_client(args)
    for game in games_from(args):
        spec = specs[game]
        cands: List[Params] = []
        seen = set()
        for rt in (args.from_run_type, args.run_type):
            for key, _m, _n, _r in cache.best(game, rt, limit=args.top):
                if key in seen:
                    continue
                seen.add(key)
                p = json.loads(key)
                if not spec.validate(p):
                    cands.append(p)
        if os.path.exists(entry_path(game)):
            with open(entry_path(game)) as fh:
                p = json.load(fh)["params"]
            if canonical(p) not in seen:
                cands.insert(0, p)
        if not cands:
            print("%s: nothing cached to verify — run `search` first." % game)
            continue
        print("\n=== verify %s at %s (%d candidates) ===" %
              (game, args.run_type, len(cands)))
        ev = Evaluator(client, cache, game, run_type=args.run_type,
                       repeats=1, workers=args.workers, budget=args.budget,
                       verbose=not args.quiet)
        finals = ev.race(cands, keep=args.keep, rounds=tuple(args.rounds))
        for p, s, n in finals:
            print("  %.2f  (n=%d)" % (s, n))
        if finals:
            best_p, best_s, best_n = finals[0]
            print("  saved -> %s" % save_entry(game, best_p, best_s,
                                               args.run_type, best_n, client.name))
        print("  %s" % ev.summary())


def cmd_best(args) -> None:
    cache = Cache(args.cache)
    total = 0.0
    bundle: Dict[str, object] = {}
    for game in games_from(args):
        path = entry_path(game)
        if not os.path.exists(path):
            print("%-18s (no entry yet)" % game)
            continue
        with open(path) as fh:
            rec = json.load(fh)
        total += rec["score"]
        bundle[game] = rec["params"]
        print("%-18s %8.2f  [%s, n=%d, %s]" %
              (game, rec["score"], rec["run_type"], rec.get("observations", 0),
               rec.get("backend", "?")))
        if args.show_params:
            print(json.dumps(rec["params"], indent=2, sort_keys=True))
    print("%-18s %8.2f / 4000" % ("TOTAL", total))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(bundle, fh, indent=2, sort_keys=True)
        print("bundle -> %s" % args.out)


def cmd_probe(args) -> None:
    """Check that every parameter name in valid_params.json is accepted.

    The published parameter tables and the example config disagree on at least
    one spelling, and a single rejected key can invalidate a whole run.
    """
    specs = load_specs()
    client = build_client(args)
    for game in games_from(args):
        spec = specs[game]
        base = spec.default()
        print("\n=== probe %s ===" % game)
        try:
            s = client.score(game, base, "fast")
            print("  full default config accepted (score %.2f)" % s)
            continue
        except RunRejected as exc:
            print("  full default REJECTED: %s" % exc)
        except ApiError as exc:
            print("  api error: %s" % exc)
            continue
        for prm in spec.params:
            one = {prm.name: base[prm.name]}
            try:
                client.score(game, one, "fast")
                print("  ok       %s" % prm.name)
            except RunRejected as exc:
                print("  REJECTED %-32s %s" % (prm.name, str(exc)[:80]))
            except ApiError as exc:
                print("  error    %-32s %s" % (prm.name, str(exc)[:80]))


def cmd_bench(args) -> None:
    """Compare optimisers on the offline mock — free, and repeatable."""
    specs = load_specs()
    client = MockClient(seed=args.seed)
    print("%-10s %-18s %10s %10s %8s" %
          ("optimizer", "game", "best(fast)", "true(full)", "calls"))
    for name in (args.optimizers or ["random", "hill", "ea", "pbil"]):
        for game in games_from(args):
            spec = specs[game]
            cache = Cache(os.path.join(args.bench_dir, "%s-%s.sqlite" % (name, game)))
            ev = Evaluator(client, cache, game, run_type="fast", repeats=1,
                           workers=args.workers, budget=args.budget, verbose=False)
            res = REGISTRY[name](spec, ev, random.Random(args.seed),
                                 iterations=10 ** 6, generations=10 ** 6)
            truth = (client.score(game, res.best_params, "full")
                     if res.best_params else float("nan"))
            print("%-10s %-18s %10.2f %10.2f %8d" %
                  (name, game, res.best_score, truth, ev.n_calls))


def cmd_status(args) -> None:
    cache = Cache(args.cache)
    stats = cache.stats()
    if not stats:
        print("cache is empty (%s)" % args.cache)
        return
    print("%-18s %8s %8s %8s" % ("game", "fast", "medium", "full"))
    for game in GAMES:
        row = stats.get(game, {})
        print("%-18s %8d %8d %8d" % (game, row.get("fast", 0),
                                     row.get("medium", 0), row.get("full", 0)))
    print("\ntotal observations: %d" %
          sum(sum(r.values()) for r in stats.values()))


# ---------------------------------------------------------------- parser
def main(argv: Optional[List[str]] = None) -> None:
    def add_common(p):
        p.add_argument("--backend", default=os.environ.get("TTB_BACKEND", "mock"),
                    choices=["mock", "local", "hosted", "modal"],
                    help="mock=offline surrogate, local=Docker API, "
                         "hosted=competition server (default: mock)")
        p.add_argument("--api-key", default="")
        p.add_argument("--local-url", default=os.environ.get("TTB_LOCAL_URL", ""))
        p.add_argument("--hosted-url", default=os.environ.get("TTB_HOSTED_URL", ""))
        p.add_argument("--timeout-ms", type=int, default=0)
        p.add_argument("--local-pool", type=int, default=1,
                       help="number of local API containers to spread across; "
                            "set --workers to the same value")
        p.add_argument("--first-port", type=int, default=3000)
        p.add_argument("--pool-hosts", default=os.environ.get("TTB_POOL_HOSTS", ""),
                       help="remote/multi-host pool, e.g. "
                            "'1.2.3.4:3000:90' or 'a:3000:90,b:3000:90'; "
                            "overrides --local-pool. Unreachable ports are pruned.")
        p.add_argument("--pool-url", default=os.environ.get("TTB_POOL_URL", ""),
                       help="single autoscaling endpoint (e.g. a Modal URL "
                            "ending in /api). Fanned out via --pool-replicas.")
        p.add_argument("--pool-replicas", type=int,
                       default=int(os.environ.get("TTB_POOL_REPLICAS", "24")),
                       help="concurrent requests to --pool-url (= worker count)")
        p.add_argument("--cache", default=os.environ.get("TTB_CACHE",
                                                      "results/cache.sqlite"))
        p.add_argument("--game", nargs="*", default=["all"])
        p.add_argument("--run-type", default="fast", choices=sorted(RUN_TYPES))
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--budget", type=int, default=None,
                    help="max API calls for this command")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--quiet", action="store_true")
    # Shared flags go on both the top-level parser and every subparser so they
    # work on either side of the subcommand. The subparser copy must carry
    # SUPPRESS defaults: otherwise argparse re-applies its defaults after the
    # subcommand and silently discards anything passed before it.
    common = argparse.ArgumentParser(add_help=False)
    add_common(common)
    sub_common = argparse.ArgumentParser(add_help=False)
    add_common(sub_common)
    for action in sub_common._actions:
        action.default = argparse.SUPPRESS

    ap = argparse.ArgumentParser(
        prog="ttbalance", parents=[common],
        description="CoG 2026 Tabletop Games Balancing Competition toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help):
        return sub.add_parser(name, help=help, parents=[sub_common])

    s = add("search", "optimise parameters for one or more games")
    s.add_argument("--optimizer", default="ea", choices=sorted(REGISTRY))
    s.add_argument("--iterations", type=int, default=200)
    s.add_argument("--generations", type=int, default=40)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--seed-from-cache", type=int, default=4,
                   help="warm-start from N best cached configs (0 disables)")
    s.add_argument("--opt-args", default="",
                   help='JSON of extra optimiser kwargs, e.g. \'{"mu":16}\'')
    s.set_defaults(func=cmd_search)

    v = add("verify", "re-race top candidates at higher fidelity")
    v.add_argument("--top", type=int, default=8)
    v.add_argument("--keep", type=int, default=3)
    v.add_argument("--from-run-type", default="fast", choices=sorted(RUN_TYPES))
    v.add_argument("--rounds", type=int, nargs="+", default=[1, 3])
    v.set_defaults(func=cmd_verify)

    b = add("best", "show current best entry per game")
    b.add_argument("--show-params", action="store_true")
    b.add_argument("--out", default="", help="write a combined JSON bundle")
    b.set_defaults(func=cmd_best)

    p = add("probe", "check parameter names against the API")
    p.set_defaults(func=cmd_probe)

    bn = add("bench", "compare optimisers offline on the mock")
    bn.add_argument("--optimizers", nargs="*", default=None)
    bn.add_argument("--bench-dir", default="results/bench")
    bn.set_defaults(func=cmd_bench)

    st = add("status", "show cache contents")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    if getattr(args, "budget", None) is None and args.cmd == "bench":
        args.budget = 1500
    args.func(args)


if __name__ == "__main__":
    main()
