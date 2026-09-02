# tabletop-game-balancer

Entry for the [2026 Tabletop Games Balancing Competition](https://balance-competition.tabletopgames.ai)
(IEEE CoG 2026), submitted as `jxucoder`.

**Best leaderboard score: 3652.5 / 4000.** The entry is the bundle in
[`results/winning_submission.json`](results/winning_submission.json).

| | |
|---|---|
| Score progression | 3490 → 3553 → 3562 → 3606 → 3607 → 3638 → 3642 → 3648 → **3652.5** |
| Real evaluations spent | ~9,000 |
| Compute cost | ~$90 (Modal) |

---

## The short version

The four games are a noisy black box: you send rule parameters, you get one
number back. Almost everything that decided our score came from how we
**measured**, not from which optimiser we ran. The optimisers here are ordinary
— PBIL, a (μ+λ) EA, and a surrogate loop around a tabular foundation model. The
three decisions that actually moved the number were:

1. **Never trust a single observation.** A `fast` run is 36 matchups. Our first
   bundle, built by picking each game's best single score, added up to 3781 and
   confirmed to 3485 — 296 points of pure luck. Everything gets raced to n≥3
   before it is believed.

2. **Check per game whether the cheap fidelity points the right way.** For
   ExplodingKittens, `fast` and `medium` rank configurations almost
   independently; a search that trusted `fast` walked backwards for hours. For
   Dominion `fast` tracks `medium` closely (fast 963 → medium 971, 961 → 969).
   Same framework, opposite conclusions — so we measured the relationship
   instead of assuming it.

3. **Change one game per submission.** Submissions are free, unlimited, and the
   only full-fidelity instrument available. One bundle that changed three games
   at once gained +45 at medium and +1.6 at full; splitting that into three
   single-variable submissions showed the entire loss was one game (Wonders7)
   and recovered +35.

---

## How the entry was produced

### 1. Evaluation throughput

The organisers ship a Docker image of the evaluator. Measured, not assumed: the
server inside it runs **one `java -jar TAG.jar` at a time** and does not queue,
so a single container gives no parallelism regardless of host cores — a 6-CPU
container finished the same job in 2352 s against 2379 s for 1 CPU. Throughput
comes from running N containers and keeping exactly one request in flight per
container.

* [`scripts/localapi.sh`](scripts/localapi.sh) — pool of N local containers.
* [`modal_localapi.py`](modal_localapi.py) — the same image on Modal, exposed as
  a **function** rather than a web endpoint: Modal's synchronous HTTP proxy
  times out around 150 s and our evaluations take up to 40 minutes.
* `LocalPoolClient` / `ModalClient` in [`src/ttbalance/client.py`](src/ttbalance/client.py)
  enforce the one-request-per-container invariant.

### 2. Noise discipline

[`src/ttbalance/evaluate.py`](src/ttbalance/evaluate.py) caches every observation
ever made (SQLite, ~9,000 rows) and never pays twice for the same work.
`Evaluator.race()` is successive halving over repeated observations: one look at
everything, then progressively more repeats on the survivors. Nothing is
submitted on fewer than three.

### 3. Fidelity diagnosis

Before trusting a cheap signal we compute, per game, how much of the spread
between configurations survives the measurement noise:

| game | spread between configs | noise sd | SNR | usable? |
|---|---|---|---|---|
| ExplodingKittens | 104.5 | 21.2 | 8.5 | yes |
| Wonders7 | 47.2 | 22.1 | 3.7 | barely |
| CantStop | 25.3 | 13.1 | 3.3 | barely |
| Dominion | 18.0 | 12.7 | 2.5 | no |

Only ExplodingKittens can be ranked reliably at `medium`. Search effort was
allocated accordingly. Reproduce with `./scripts/reproduce.sh diagnose`.

### 4. Search

[`src/ttbalance/optimizers/`](src/ttbalance/optimizers/) holds four:
`random` and `hill` as controls, `pbil` (population-based incremental learning,
which averages over a whole generation and so resists noise), and `nori` — a
surrogate loop around [Synthefy's Nori](https://github.com/Synthefy/synthefy-nori)
tabular foundation model:

1. Fit Nori in-context on every measured configuration (X = encoded parameters,
   y = mean `medium` score). Only `medium` — feeding it `fast` for
   ExplodingKittens teaches it the wrong signal.
2. Propose ~2,000 unseen candidates; predict all of them in one pass.
3. Rank by `q50 + κ·(q90 − q50)` — Nori's quantiles are what make the explore
   term possible.
4. Evaluate only the top ~10 for real, fold in, repeat.

It also carries a TuRBO-style adaptive trust region, greedy local-penalisation
batch diversity, and incumbent re-observation.

**Attribution:** three of the four configurations in the final bundle have zero
`fast` observations anywhere in our records — they were proposed directly by the
surrogate at `medium`, not promoted from a cheap search.

### 5. Multi-fidelity

[`src/ttbalance/multifidelity.py`](src/ttbalance/multifidelity.py) is the lesson
from (2) turned into a model. Ranking by `fast` fails, but that is an argument
against `fast` as a *ranker*, not as a *feature*: train on
`[params, fast_score] → medium_score`, with NaN where a configuration was never
run cheaply. Leave-one-out on the 18 ExplodingKittens configurations carrying
both fidelities: MAE **28.21 → 25.97** (mean baseline 35.10). The intended loop —
fast-screen a wide pool, spend `medium` only on what the model ranks highest —
is [`scripts/mf_screen.py`](scripts/mf_screen.py).

### 6. Reading the components

TAG logs more than the score. `modal_localapi.py` parses the server's stdout and
returns `matrix_distance`, `fpa_diff`, and the realised and target win-rate
matrices alongside the number. For our best ExplodingKittens configuration the
entire loss is matrix distance (190) with `fpa_diff` at 0, and the largest
single miss is **Elite beating Good 33.3% against a 60% target** — our
optimisation had been flattening the skill gradient when the target wanted it
steeper. The current season's target matrices are not published anywhere, so
this readout is the only way to see them.

We found this late and could not act on it: the follow-up sweeps were run at
`fast`, whose conclusions `medium` then rejected — the same trap as (2). It is
the clearest direction for a future entry.

---

## Reproducing

```bash
./scripts/reproduce.sh check       # offline: install, tests, optimiser on the built-in surrogate
./scripts/reproduce.sh evaluator   # start 10 local API containers (needs Docker)
./scripts/reproduce.sh diagnose    # per-game fast-vs-medium signal-to-noise
./scripts/reproduce.sh search      # medium-direct search, one loop per game
./scripts/reproduce.sh confirm     # race the leaders to n>=3
./scripts/reproduce.sh bundle      # print the submission JSON
```

`check` needs nothing but Python and finishes in about a minute — it runs the
test suite and a full optimiser loop against `src/ttbalance/mock.py`, an offline
surrogate of the API. The later stages need a real evaluator and take hours.

For Modal instead of local Docker: `modal deploy modal_localapi.py`, then pass
`--backend modal`.

---

## Layout

```
src/ttbalance/
  spec.py           parameter space, mutation, neighbourhoods
  client.py         Local / LocalPool / Hosted / Modal evaluators
  cache.py          every observation ever made (SQLite)
  evaluate.py       budgeted parallel evaluation + successive-halving racing
  encode.py         configuration -> fixed-width numeric row
  multifidelity.py  cheap score as a feature for predicting the expensive one
  transfer.py       per-game medium->full transfer rates from submission history
  calibrate.py      medium-estimate -> full-score calibration
  mock.py           offline surrogate, so the pipeline runs with no API
  optimizers/       random, hill, pbil, ea, nori
scripts/            evaluator pool, search loops, confirmation, screening
modal_localapi.py   the evaluator on Modal, with score-component readout
docs/COMPETITION.md task, API, scoring, deadlines
results/winning_submission.json   the submitted bundle
```

31 tests: `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

---

## What did not work

Recorded because the negative results carry most of the information.

* **Ranking by `fast`.** Cost hours of search walking the wrong way for
  ExplodingKittens. The fix was per-game diagnosis, not a better optimiser.
* **Bundling several games into one submission.** Transfer collapsed to 0.13
  because one component was inflated by an n=1 estimate. Single-variable
  submissions transferred near 1.0.
* **Trusting the matrix diagnostic at `fast`.** Twice produced a
  convincing-looking direction (`SEETHEFUTURE=6`, `NOPE=10`, each halving matrix
  distance) that `medium` then rejected. The diagnostic is sound; the fidelity
  it was read at was not.
* **A variance heuristic** — that truly balanced configurations should show
  higher variance across repeats, since their measured distance is mostly noise.
  Tested against two configurations with known full scores; refuted (sd 15.3 vs
  16.3, the wrong way round).

## Acknowledgements

[TAG framework](https://github.com/GAIGResearch/TabletopGames) and the
competition organisers for shipping a local evaluator image — the entire
approach depends on being able to run evaluations off the hosted queue.
