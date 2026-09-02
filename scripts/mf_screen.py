"""Cheap parallel screening: fast-screen a wide pool, medium-verify the few.

A fast run costs ~2 min (~$0.007); a medium one 15-40 min (~$0.05-0.10). So we
buy breadth at fast and depth at medium: score a large candidate pool cheaply
in parallel on Modal, feed those scores to the multi-fidelity model as a
feature (never as a ranker — fast tracks medium only weakly), and spend medium
runs on the handful the model ranks highest.
"""
import json, sys, time
sys.path.insert(0, 'src')
from concurrent.futures import ThreadPoolExecutor
import modal
from ttbalance.cache import Cache
from ttbalance.multifidelity import MultiFidelityModel
from ttbalance.spec import canonical, load_specs

GAME = sys.argv[1] if len(sys.argv) > 1 else "ExplodingKittens"
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WORKERS = 40

spec = load_specs()[GAME]
cache = Cache("results/cache.sqlite")
f = modal.Cls.from_name("ttb-localapi", "Evaluator")()
best = json.load(open('results/final_best.json'))
import random
rng = random.Random(11)

# Pool: mutations of the leaders plus fresh random draws.
leaders = [json.loads(r[0]) for r in cache.best(GAME, "medium", limit=12, min_obs=2)]
leaders = [p for p in leaders if not spec.validate(p)] or [best[GAME]]
seen = set()
pool = []
while len(pool) < POOL and len(seen) < POOL * 20:
    p = spec.mutate(rng.choice(leaders), rng, rate=0.25) if rng.random() < 0.7 else spec.sample(rng)
    k = canonical(p)
    if k in seen or spec.validate(p):
        continue
    seen.add(k); pool.append(p)
print("pool %d" % len(pool), flush=True)

t0 = time.time()
def fast_one(p):
    try:
        r = f.run.remote(GAME, p, "fast", 900000)
        s = r.get("score")
        if s:
            cache.add(GAME, p, "fast", s)
        return s
    except Exception:
        return None
with ThreadPoolExecutor(WORKERS) as ex:
    fast_scores = list(ex.map(fast_one, pool))
ok = [(p, s) for p, s in zip(pool, fast_scores) if s]
print("fast screened %d/%d in %.0fs" % (len(ok), len(pool), time.time() - t0), flush=True)

m = MultiFidelityModel()
if not m.fit(cache, spec, GAME):
    print("not enough expensive data to fit"); raise SystemExit
print("fitted on %d rows (%d with cheap feature)" % (m.n_train, m.n_with_cheap), flush=True)

acq = m.acquisition(spec, [p for p, _ in ok], [s for _, s in ok])
order = sorted(range(len(ok)), key=lambda i: -acq[i])[:TOPK]
picks = [ok[i][0] for i in order]
print("top-%d predicted: %s" % (TOPK, ", ".join("%.1f" % acq[i] for i in order)), flush=True)

def med_one(p):
    out = []
    for _ in range(3):
        try:
            r = f.run.remote(GAME, p, "medium", 3000000)
            s = r.get("score")
            if s:
                cache.add(GAME, p, "medium", s); out.append(s)
        except Exception:
            pass
    return (p, sum(out) / len(out) if out else 0)
with ThreadPoolExecutor(min(WORKERS, TOPK)) as ex:
    res = list(ex.map(med_one, picks))
res.sort(key=lambda t: -t[1])
print("\n=== medium confirmed (n=3) ===", flush=True)
for p, s in res:
    print("  %.1f" % s, flush=True)
json.dump(res[0][0], open('results/mf_best_%s.json' % GAME, 'w'))
print("BEST %.1f  (incumbent EK 895.4)" % res[0][1], flush=True)
