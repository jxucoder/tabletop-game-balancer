"""Regression tests. Run: PYTHONPATH=src python3 -m unittest discover -s tests"""
from __future__ import annotations

import json
import time
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from ttbalance import cli
from ttbalance.cache import Cache
from ttbalance.client import RunRejected
from ttbalance.evaluate import BudgetExhausted, Evaluator
from ttbalance.mock import MockClient
from ttbalance.optimizers import REGISTRY
from ttbalance.spec import GAMES, Subset, canonical, load_specs

SPECS = load_specs()


def tmp_cache() -> Cache:
    return Cache(os.path.join(tempfile.mkdtemp(), "c.sqlite"))


class TestSpec(unittest.TestCase):
    def test_defaults_and_samples_are_valid(self):
        rng = random.Random(0)
        for game, spec in SPECS.items():
            self.assertEqual(spec.validate(spec.default()), [], game)
            for _ in range(50):
                self.assertEqual(spec.validate(spec.sample(rng)), [], game)

    def test_mutation_and_crossover_stay_valid(self):
        rng = random.Random(1)
        for game, spec in SPECS.items():
            a, b = spec.sample(rng), spec.sample(rng)
            for _ in range(50):
                child = spec.mutate(spec.crossover(a, b, rng), rng, rate=0.4)
                self.assertEqual(spec.validate(child), [], game)

    def test_mutation_always_changes_something(self):
        rng = random.Random(2)
        spec = SPECS["CantStop"]
        base = spec.default()
        for _ in range(50):
            self.assertNotEqual(canonical(spec.mutate(base, rng, rate=0.0)),
                                canonical(base))

    def test_neighbours_are_valid_and_single_step(self):
        for game, spec in SPECS.items():
            base = spec.default()
            for nb in spec.neighbours(base):
                self.assertEqual(spec.validate(nb), [], game)
                diff = [k for k in base if canonical({k: base[k]}) !=
                        canonical({k: nb[k]})]
                self.assertEqual(len(diff), 1, game)

    def test_subset_sizes_respected(self):
        rng = random.Random(3)
        for game, spec in SPECS.items():
            for prm in spec.params:
                if not isinstance(prm, Subset):
                    continue
                for _ in range(30):
                    v = prm.sample(rng)
                    self.assertTrue(prm.min_size <= len(v) <= prm.max_size)
                    self.assertEqual(len(set(v)), len(v))

    def test_canonical_ignores_list_order(self):
        self.assertEqual(canonical({"CARDS": ["B", "A"]}),
                         canonical({"CARDS": ["A", "B"]}))


class TestMock(unittest.TestCase):
    def test_optimum_scores_near_maximum(self):
        m = MockClient(seed=5)
        for game in GAMES:
            self.assertGreater(m.score(game, m.optimum(game), "full"), 950)

    def test_invalid_params_rejected(self):
        m = MockClient(seed=5)
        bad = dict(SPECS["CantStop"].default())
        bad["MARKERS"] = 99
        with self.assertRaises(RunRejected):
            m.score("CantStop", bad, "fast")

    def test_higher_fidelity_is_less_noisy(self):
        m = MockClient(seed=5)
        p = SPECS["CantStop"].default()
        spread = {}
        for rt in ("fast", "full"):
            xs = [m.score("CantStop", p, rt) for _ in range(60)]
            mean = sum(xs) / len(xs)
            spread[rt] = sum((x - mean) ** 2 for x in xs) / len(xs)
        self.assertLess(spread["full"], spread["fast"])


class TestEvaluator(unittest.TestCase):
    def test_cache_prevents_repeat_api_calls(self):
        ev = Evaluator(MockClient(0), tmp_cache(), "CantStop", workers=1,
                       verbose=False)
        p = SPECS["CantStop"].default()
        ev.evaluate(p)
        ev.evaluate(p)
        self.assertEqual(ev.n_calls, 1)

    def test_repeats_accumulate_observations(self):
        cache = tmp_cache()
        ev = Evaluator(MockClient(0), cache, "CantStop", workers=1, verbose=False)
        p = SPECS["CantStop"].default()
        ev.evaluate(p, repeats=5)
        self.assertEqual(len(cache.scores("CantStop", p, "fast")), 5)
        self.assertEqual(ev.n_calls, 5)

    def test_budget_is_never_exceeded(self):
        rng = random.Random(0)
        spec = SPECS["CantStop"]
        ev = Evaluator(MockClient(0), tmp_cache(), "CantStop", workers=4,
                       budget=25, verbose=False)
        ev.evaluate_many([spec.sample(rng) for _ in range(200)])
        self.assertLessEqual(ev.n_calls, 25)
        self.assertTrue(ev.exhausted)
        with self.assertRaises(BudgetExhausted):
            ev.check()

    def test_race_returns_multi_observation_estimates(self):
        rng = random.Random(0)
        spec = SPECS["CantStop"]
        ev = Evaluator(MockClient(0), tmp_cache(), "CantStop", workers=4,
                       verbose=False)
        finals = ev.race([spec.sample(rng) for _ in range(8)], keep=2,
                         rounds=(1, 3))
        self.assertEqual(len(finals), 2)
        for _p, _s, n in finals:
            self.assertGreaterEqual(n, 3)


class TestOptimizers(unittest.TestCase):
    @staticmethod
    def _runnable_optimizers():
        """Skip optimisers whose optional dependencies are absent: `nori` needs
        numpy and the Nori SDK, and a missing extra should not fail the suite."""
        out = {}
        for name, fn in REGISTRY.items():
            if name == "nori":
                try:
                    import numpy  # noqa: F401
                    import synthefy_nori  # noqa: F401
                except Exception:
                    continue
            out[name] = fn
        return out

    def test_every_optimizer_beats_a_random_baseline(self):
        client = MockClient(seed=11)
        spec = SPECS["CantStop"]
        rng = random.Random(0)
        baseline = max(client.score("CantStop", spec.sample(rng), "full")
                       for _ in range(30))
        for name, fn in self._runnable_optimizers().items():
            ev = Evaluator(client, tmp_cache(), "CantStop", workers=4,
                           budget=400, verbose=False)
            res = fn(spec, ev, random.Random(0), iterations=10 ** 6,
                     generations=10 ** 6)
            self.assertTrue(res.best_params, name)
            self.assertEqual(spec.validate(res.best_params), [], name)
            self.assertLessEqual(ev.n_calls, 400, name)
            truth = client.score("CantStop", res.best_params, "full")
            self.assertGreater(truth, baseline, "%s: %.1f <= %.1f"
                               % (name, truth, baseline))

    def test_optimizers_terminate_under_tiny_budget(self):
        client = MockClient(seed=3)
        spec = SPECS["ExplodingKittens"]
        for name, fn in self._runnable_optimizers().items():
            ev = Evaluator(client, tmp_cache(), "ExplodingKittens", workers=2,
                           budget=12, verbose=False)
            fn(spec, ev, random.Random(0), iterations=10 ** 6,
               generations=10 ** 6)
            self.assertLessEqual(ev.n_calls, 12, name)


class TestCli(unittest.TestCase):
    def _parsed(self, argv):
        seen = {}
        orig = cli.cmd_search
        cli.cmd_search = lambda a: seen.setdefault("a", a)
        try:
            cli.main(argv)
        finally:
            cli.cmd_search = orig
        return seen["a"]

    def test_common_flags_work_before_the_subcommand(self):
        a = self._parsed(["--budget", "77", "--quiet", "--backend", "local",
                          "search"])
        self.assertEqual(a.budget, 77)
        self.assertTrue(a.quiet)
        self.assertEqual(a.backend, "local")

    def test_common_flags_work_after_the_subcommand(self):
        a = self._parsed(["search", "--budget", "77", "--backend", "local"])
        self.assertEqual(a.budget, 77)
        self.assertEqual(a.backend, "local")

    def test_subcommand_flags_still_parse(self):
        a = self._parsed(["--workers", "3", "search", "--optimizer", "pbil"])
        self.assertEqual(a.workers, 3)
        self.assertEqual(a.optimizer, "pbil")


class TestEntries(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)

    def _read(self):
        with open(cli.entry_path("CantStop")) as fh:
            return json.load(fh)

    def test_higher_fidelity_wins_even_with_a_lower_score(self):
        p = SPECS["CantStop"].default()
        cli.save_entry("CantStop", p, 999.0, "fast", 9, "mock")
        cli.save_entry("CantStop", p, 900.0, "medium", 3, "mock")
        rec = self._read()
        self.assertEqual(rec["run_type"], "medium")
        self.assertEqual(rec["score"], 900.0)

    def test_lower_fidelity_never_downgrades_an_entry(self):
        p = SPECS["CantStop"].default()
        cli.save_entry("CantStop", p, 900.0, "medium", 3, "mock")
        cli.save_entry("CantStop", p, 999.0, "fast", 50, "mock")
        self.assertEqual(self._read()["run_type"], "medium")

    def test_equal_fidelity_keeps_the_better_entry(self):
        p = SPECS["CantStop"].default()
        cli.save_entry("CantStop", p, 900.0, "medium", 5, "mock")
        cli.save_entry("CantStop", p, 800.0, "medium", 5, "mock")
        self.assertEqual(self._read()["score"], 900.0)

    def test_pick_entry_prefers_a_repeatedly_observed_mean(self):
        cache = tmp_cache()
        spec = SPECS["CantStop"]
        lucky = spec.default()
        steady = spec.mutate(lucky, random.Random(0), rate=1.0)
        cache.add("CantStop", lucky, "fast", 1000.0)          # one lucky draw
        for _ in range(5):
            cache.add("CantStop", steady, "fast", 950.0)      # consistent
        params, mean, n = cli.pick_entry(cache, spec, "CantStop", "fast", lucky)
        self.assertEqual(canonical(params), canonical(steady))
        self.assertAlmostEqual(mean, 950.0)
        self.assertEqual(n, 5)


if __name__ == "__main__":
    unittest.main()


class TestHostedPolling(unittest.TestCase):
    """The server reports `created` before a worker picks a run up; an unknown
    status must not abort a long unattended search."""

    def _client(self, statuses):
        from ttbalance.client import HostedClient
        c = HostedClient("key", poll_interval=0.0)
        c.submit = lambda *a, **k: 1
        seq = list(statuses)
        c.status = lambda rid: seq.pop(0)
        c.result = lambda rid: 900.0
        return c

    def test_created_status_is_pending_not_an_error(self):
        c = self._client(["created", "created", "running", "complete"])
        self.assertEqual(c.score("CantStop", {}, "fast"), 900.0)

    def test_unknown_status_is_tolerated(self):
        c = self._client(["surprise", "running", "complete"])
        self.assertEqual(c.score("CantStop", {}, "fast"), 900.0)

    def test_failure_status_raises(self):
        from ttbalance.client import ApiError
        c = self._client(["running", "failed"])
        with self.assertRaises(ApiError):
            c.score("CantStop", {}, "fast")


class TestPbilState(unittest.TestCase):
    """Short repeated passes must compound, not restart from uniform."""

    def test_marginals_survive_a_restart(self):
        from ttbalance.optimizers.pbil import Marginals, optimize
        spec = SPECS["CantStop"]
        path = os.path.join(tempfile.mkdtemp(), "pbil_CantStop.json")
        client = MockClient(seed=4)

        ev = Evaluator(client, tmp_cache(), "CantStop", workers=4, budget=200,
                       verbose=False)
        optimize(spec, ev, random.Random(0), generations=10 ** 6,
                 state_path=path)
        self.assertTrue(os.path.exists(path))

        first = Marginals(spec)
        self.assertTrue(first.load(path))
        uniform = Marginals(spec)
        self.assertNotEqual(first.p, uniform.p)

        # A second pass resumes rather than starting over.
        ev2 = Evaluator(client, tmp_cache(), "CantStop", workers=4, budget=200,
                        verbose=False)
        optimize(spec, ev2, random.Random(1), generations=10 ** 6,
                 state_path=path)
        second = Marginals(spec)
        self.assertTrue(second.load(path))
        self.assertNotEqual(second.p, first.p)

    def test_load_rejects_a_mismatched_space(self):
        from ttbalance.optimizers.pbil import Marginals
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        Marginals(SPECS["CantStop"]).save(path)
        self.assertFalse(Marginals(SPECS["Dominion"]).load(path))

    def test_load_tolerates_a_corrupt_file(self):
        from ttbalance.optimizers.pbil import Marginals
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        with open(path, "w") as fh:
            fh.write("{not json")
        self.assertFalse(Marginals(SPECS["CantStop"]).load(path))


class TestLocalPool(unittest.TestCase):
    """Each container serialises its work, so the pool must never issue two
    concurrent requests to the same endpoint."""

    def test_one_request_in_flight_per_endpoint(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from ttbalance.client import LocalPoolClient, pool_urls

        pool = LocalPoolClient(pool_urls(4))
        active = {c.base_url: 0 for c in pool.clients}
        lock = threading.Lock()
        breach = []

        def fake(self, game, params, run_type="fast"):
            with lock:
                active[self.base_url] += 1
                if active[self.base_url] > 1:
                    breach.append(self.base_url)
            time.sleep(0.01)
            with lock:
                active[self.base_url] -= 1
            return 1.0

        for c in pool.clients:
            c.score = fake.__get__(c)
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda _: pool.score("CantStop", {}, "fast"), range(40)))
        self.assertEqual(breach, [])

    def test_endpoint_returned_to_pool_after_failure(self):
        from ttbalance.client import ApiError, LocalPoolClient, pool_urls
        pool = LocalPoolClient(pool_urls(2))
        for c in pool.clients:
            c.score = lambda *a, **k: (_ for _ in ()).throw(ApiError("boom"))
        for _ in range(5):
            with self.assertRaises(ApiError):
                pool.score("CantStop", {}, "fast")
        self.assertEqual(pool._free.qsize(), 2)   # not leaked

    def test_pool_urls_are_distinct_ports(self):
        from ttbalance.client import pool_urls
        urls = pool_urls(12, 3000)
        self.assertEqual(len(set(urls)), 12)
        self.assertIn("http://localhost:3011/api/", urls)


class TestHostedResilience(unittest.TestCase):
    """A dropped connection must become a ret=>ApiError, never an uncaught
    traceback that kills an unattended search."""

    def _client(self):
        from ttbalance.client import HostedClient
        c = HostedClient("key", retries=3)
        return c

    def test_connection_reset_is_wrapped_and_retried(self):
        import requests
        from ttbalance.client import ApiError
        c = self._client()
        calls = {"n": 0}

        def boom(method, url, **kw):
            calls["n"] += 1
            raise requests.exceptions.ConnectionError("reset by peer")

        c._session.request = boom
        # patch sleep so the test is instant
        import ttbalance.client as mod
        orig = mod.time.sleep
        mod.time.sleep = lambda *_: None
        try:
            with self.assertRaises(ApiError):
                c.status(1)
        finally:
            mod.time.sleep = orig
        self.assertEqual(calls["n"], 3)   # retried, not crashed

    def test_transient_reset_then_success(self):
        import requests

        class Resp:
            status_code = 200
            def json(self): return {"run_status": "complete"}

        c = self._client()
        seq = [requests.exceptions.ConnectionError("x"), Resp()]

        def flaky(method, url, **kw):
            v = seq.pop(0)
            if isinstance(v, Exception):
                raise v
            return v

        c._session.request = flaky
        import ttbalance.client as mod
        orig = mod.time.sleep
        mod.time.sleep = lambda *_: None
        try:
            self.assertEqual(c.status(1), "complete")
        finally:
            mod.time.sleep = orig


class TestMultiHostPool(unittest.TestCase):
    def test_parse_pool_spec_single_and_multi(self):
        from ttbalance.client import parse_pool_spec
        self.assertEqual(parse_pool_spec("localhost:3000:2"),
                         ["http://localhost:3000/api/", "http://localhost:3001/api/"])
        u = parse_pool_spec("1.2.3.4:3000:2, 5.6.7.8:3010:1")
        self.assertEqual(u, ["http://1.2.3.4:3000/api/", "http://1.2.3.4:3001/api/",
                             "http://5.6.7.8:3010/api/"])

    def test_prune_drops_unreachable(self):
        from ttbalance.client import LocalPoolClient
        # one obviously-dead endpoint; prune should drop it and keep none-or-raise
        good = ["http://127.0.0.1:59999/api/"]  # nothing listening
        with self.assertRaises(Exception):
            LocalPoolClient(good, prune=True, prune_timeout=0.3)


try:
    import numpy  # noqa: F401
    import synthefy_nori  # noqa: F401
    _HAVE_NORI = True
except Exception:
    _HAVE_NORI = False


@unittest.skipUnless(_HAVE_NORI, "needs numpy + synthefy_nori (.venv-nori)")
class TestSurrogateUpgrades(unittest.TestCase):
    """The three surrogate.py upgrades: re-observation depth, batch diversity
    via local penalisation, and the trust-region mutation rate."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from ttbalance.optimizers.surrogate import optimize
        cls.cache = tmp_cache()
        cls.ev = Evaluator(MockClient(seed=7), cls.cache, "CantStop",
                           workers=2, budget=80, verbose=False)
        cls.res = optimize(SPECS["CantStop"], cls.ev, random.Random(0),
                           generations=3, pool=300, batch=6, init=15,
                           reobserve=2)

    # -- upgrade 1: re-observation of thin incumbents -------------------
    def test_reobserve_builds_observation_depth(self):
        # The mechanism actually ran (bootstrap leaves 15 configs at n=1, so
        # generation 1 always finds thin incumbents to re-observe) ...
        self.assertGreaterEqual(self.res.meta["reobserved"], 1)
        # ... and the cache ends up with real confirmation depth: at least
        # one config measured 2+ times at the search fidelity.
        deep = self.cache.best("CantStop", "fast", limit=100000, min_obs=2)
        self.assertTrue(deep)

    def test_reobserve_adds_depth_even_at_pick_repeats_1(self):
        # Production medium runs pass pick_repeats=1 (search_modal_medium.sh).
        # observe() early-returns once `repeats` scores are cached, so if the
        # incumbents merely rode along in the exploration batch, re-observation
        # would be a silent no-op in exactly the deployment it was built for.
        # The final race is disabled here, so any observation depth in the
        # cache can only have come from the re-observation mechanism.
        from ttbalance.optimizers.surrogate import optimize
        cache = tmp_cache()
        ev = Evaluator(MockClient(seed=11), cache, "CantStop",
                       workers=2, budget=60, verbose=False)
        ev.race = lambda *a, **k: []   # depth must come from reobserve alone
        res = optimize(SPECS["CantStop"], ev, random.Random(1),
                       generations=2, pool=200, batch=4, init=10,
                       pick_repeats=1, reobserve=2)
        self.assertGreaterEqual(res.meta["reobserved"], 1)
        deep = cache.best("CantStop", "fast", limit=100000, min_obs=2)
        self.assertTrue(deep)
        # Every reported re-observation consumed a real API call on top of
        # the bootstrap (10) and exploration picks (2 gens x 4 x 1 repeat):
        # the meta counter reports observations that actually happened.
        self.assertGreaterEqual(ev.n_calls,
                                10 + 2 * 4 + res.meta["reobserved"])

    def test_budget_respected_and_result_valid(self):
        self.assertLessEqual(self.ev.n_calls, 80)
        self.assertTrue(self.res.best_params)
        self.assertEqual(SPECS["CantStop"].validate(self.res.best_params), [])

    # -- upgrade 2: batch diversity (greedy local penalisation) ---------
    def test_identical_candidates_cannot_both_win_at_full_acq(self):
        import numpy as np
        from ttbalance.optimizers.surrogate import _select_batch
        Xc = np.array([[1.0, 2.0, 3.0, 4.0],
                       [1.0, 2.0, 3.0, 4.0],    # exact duplicate of row 0
                       [9.0, 8.0, 7.0, 6.0]])   # distinct point
        acq = np.array([10.0, 10.0, 6.0])
        order = _select_batch(acq, Xc, batch=2)
        # Row 0 wins the first argmax; the duplicate row 1 is then penalised
        # (10 -> 5 < 6), so the distinct row 2 must be picked second even
        # though row 1's unpenalised acquisition ties for the maximum.
        self.assertEqual(order, [0, 2])

    def test_selection_never_repicks_and_fills_the_batch(self):
        import numpy as np
        from ttbalance.optimizers.surrogate import _select_batch
        Xc = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        order = _select_batch(np.array([5.0, 5.0, 1.0]), Xc, batch=5)
        self.assertEqual(sorted(order), [0, 1, 2])   # all distinct, no repeat

    # -- upgrade 3: trust-region mutation rate --------------------------
    def test_mutation_rate_adapts_within_bounds(self):
        rate = self.res.meta["mut_rate"]
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.05)
        self.assertLessEqual(rate, 0.5)


class TestMultiFidelity(unittest.TestCase):
    """The cheap-fidelity column must be optional per row: configs never run
    cheaply still train the model on everything their parameters say."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("numpy/pandas not installed")
        from ttbalance.multifidelity import build_queries, build_training
        self.build_training = build_training
        self.build_queries = build_queries

    def test_training_frame_marks_unmeasured_cheap_as_nan(self):
        import math
        cache = tmp_cache()
        spec = SPECS["CantStop"]
        rng = random.Random(3)
        both, expensive_only = spec.sample(rng), spec.sample(rng)
        cache.add("CantStop", both, "fast", 900.0)
        cache.add("CantStop", both, "medium", 910.0)
        cache.add("CantStop", expensive_only, "medium", 880.0)

        X, y = self.build_training(cache, spec, "CantStop")
        self.assertEqual(len(y), 2)                      # only medium rows
        cheap = sorted(X["cheap_score"].tolist(), key=lambda v: math.isnan(v))
        self.assertAlmostEqual(cheap[0], 900.0)
        self.assertTrue(math.isnan(cheap[1]))            # missing, not zero

    def test_expensive_only_cache_still_builds(self):
        import math
        cache = tmp_cache()
        spec = SPECS["CantStop"]
        rng = random.Random(4)
        for _ in range(3):
            cache.add("CantStop", spec.sample(rng), "medium", 900.0)
        X, y = self.build_training(cache, spec, "CantStop")
        self.assertEqual(len(y), 3)
        self.assertTrue(all(math.isnan(v) for v in X["cheap_score"]))

    def test_query_frame_columns_match_training(self):
        cache = tmp_cache()
        spec = SPECS["CantStop"]
        rng = random.Random(5)
        cache.add("CantStop", spec.sample(rng), "medium", 900.0)
        X, _ = self.build_training(cache, spec, "CantStop")
        Q = self.build_queries(spec, [spec.sample(rng) for _ in range(4)],
                               [880.0, None, 895.0, None])
        self.assertEqual(list(X.columns), list(Q.columns))
        self.assertEqual(len(Q), 4)
