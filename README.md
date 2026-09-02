# tabletop-game-balancer

Entry for the [2026 Tabletop Games Balancing Competition](https://balance-competition.tabletopgames.ai)
(IEEE CoG 2026). **Best leaderboard score: 3652.5 / 4000**, from ~9,000
evaluations. The submitted bundle is [`results/winning_submission.json`](results/winning_submission.json).

Four games (Dominion, Exploding Kittens, 7 Wonders, Can't Stop) are a noisy
black box: send rule parameters, get one number back. The optimisers used here
are ordinary. What decided the score was **measurement**.

---

## Methods

### 1. Confirm before believing

A `fast` run is 36 matchups, so a single score is mostly noise. A bundle built
by taking each game's best single observation read **3781** and confirmed to
**3485** — 296 points of luck.

[`evaluate.py`](src/ttbalance/evaluate.py) caches every observation ever made
and never pays twice. `Evaluator.race()` runs successive halving over repeats:
one look at everything, more repeats on the survivors. Nothing is submitted
below n≥3.

### 2. Diagnose the cheap fidelity per game

Whether `fast` can rank configurations is a per-game empirical question, not an
assumption. For Dominion it can (fast 963 → medium 971, 961 → 969). For
Exploding Kittens it effectively cannot, and a `fast`-driven search moved
backwards for hours before this was measured.

The test is whether the spread between configurations survives the noise:

| game | spread | noise sd | SNR | rankable at `medium`? |
|---|---|---|---|---|
| Exploding Kittens | 104.5 | 21.2 | 8.5 | yes |
| 7 Wonders | 47.2 | 22.1 | 3.7 | barely |
| Can't Stop | 25.3 | 13.1 | 3.3 | barely |
| Dominion | 18.0 | 12.7 | 2.5 | no |

Search effort was allocated accordingly. `./scripts/reproduce.sh diagnose`.

### 3. One variable per submission

Submissions are unlimited and are the only full-fidelity instrument available,
so each one is an experiment — and only a controlled one is informative.

A bundle changing three games at once gained **+45 at medium, +1.6 at full**.
Split into single-variable submissions, the entire loss localised to one game
(7 Wonders) and **+35** was recovered. [`transfer.py`](src/ttbalance/transfer.py)
derives per-game transfer rates from submission history: single-game changes off
a well-observed baseline transfer near 1.0; bundles built on thin observations
transfer at 0.13.

### 4. Surrogate search

[`optimizers/`](src/ttbalance/optimizers/) holds `random` and `hill` as controls,
`pbil` (averages over a generation, so it resists noise), and `nori` — a loop
around [Synthefy's Nori](https://github.com/Synthefy/synthefy-nori) tabular
foundation model:

1. Fit in-context on measured configurations (X = encoded parameters, y = mean
   `medium` score). Only `medium` — `fast` labels teach the wrong signal.
2. Propose ~2,000 unseen candidates, predict in one pass.
3. Rank by `q50 + κ·(q90 − q50)`; the quantiles are what make the explore term
   possible.
4. Evaluate the top ~10, fold in, repeat.

Plus a TuRBO-style adaptive trust region, greedy local-penalisation batch
diversity, and incumbent re-observation.

Three of the four configurations in the final bundle have **zero `fast`
observations** — proposed directly by the surrogate at `medium`, not promoted
from a cheap search.

<details>
<summary><b>Why a tabular foundation model, and where it falls short</b></summary>

Nori does regression by in-context learning: hand it labelled rows and it
predicts new ones in a single forward pass, with no training step and no
hyperparameters to fit. Four properties made it a good fit for this problem:

* **Refits for free.** The surrogate is rebuilt every generation on a growing
  dataset. A Gaussian process needs a kernel choice and a hyperparameter fit
  each time; Nori takes the table as it is.
* **Mixed, structured features.** Our encoding is 13–36 columns mixing ordered
  integers with multi-hot card and wonder selections. That wants a bespoke
  kernel from a GP; a tabular model just reads the columns.
* **Missing values are first-class.** This is what makes the multi-fidelity
  design in §5 work at all: configurations never run at `fast` carry NaN in that
  column rather than a fabricated number.
* **Calibrated quantiles.** `q90 − q50` is the explore term in the acquisition
  function. A point regressor cannot say where it is guessing, so it cannot tell
  you where exploring pays.

The regime also suits it — 20 to 750 labelled rows per game is far too few to
train a network from scratch, and the model's synthetic-data prior does the
regularising.

The costs are real:

* **No notion of label reliability.** Every row is weighted equally whether its
  label is a mean of one observation or of seven. Our labels are means of
  varying `n` over a very noisy objective, which is exactly the heteroscedastic
  case classical BO models explicitly and this does not.
* **Confidently wrong on a bad signal.** Fed `fast` labels for Exploding
  Kittens it learned the misleading relationship faithfully and pointed the
  search at bad regions with tight quantiles. A GP with an explicit noise term
  would at least have widened its posterior. The model cannot rescue a
  measurement mistake — see §2.
* **Opaque.** No kernel to inspect, no coefficients to read. Questions like
  "which parameter actually matters" had to be answered by separate analysis,
  not by interrogating the surrogate.
* **Wrong tool below ~15 labels.** The `medium → full` calibration has only one
  label per submission, so [`calibrate.py`](src/ttbalance/calibrate.py) and
  [`transfer.py`](src/ttbalance/transfer.py) deliberately use plain regression
  there instead; a foundation model would overfit a dozen points.
* **Unproven margin.** No controlled A/B was run against `pbil` on the same
  `medium` data, so the honest claim is that these configurations came out of
  the Nori loop — not that another optimiser would have missed them.

</details>

### 5. Cheap score as a feature, not a ranker

Ranking by `fast` fails; that argues against `fast` as a *ranker*, not as a
*feature*. [`multifidelity.py`](src/ttbalance/multifidelity.py) trains on
`[params, fast_score] → medium_score`, NaN where a configuration was never run
cheaply. Leave-one-out on the 18 Exploding Kittens configurations carrying both
fidelities: **MAE 28.21 → 25.97** (mean baseline 35.10).
[`scripts/mf_screen.py`](scripts/mf_screen.py) implements the loop: screen wide
at `fast`, spend `medium` on what the model ranks highest.

### 6. Read the score's components

TAG logs more than the score. [`modal_localapi.py`](modal_localapi.py) parses
the server's stdout for `matrix_distance`, `fpa_diff`, and the realised and
target win-rate matrices. For the best Exploding Kittens configuration the
entire loss is matrix distance (190) with `fpa_diff` at 0, and the largest
single miss is **Elite beating Good 33.3% against a 60% target** — the search
had been flattening the skill gradient where the target wanted it steeper. The
season's target matrices are not published, so this readout is the only way to
see them.

### 7. Evaluation throughput

Measured, not assumed: the evaluator image runs **one `java -jar TAG.jar` at a
time** and does not queue, so one container gives no parallelism regardless of
cores — a 6-CPU container finished the same job in 2352 s against 2379 s for
1 CPU. Throughput needs N containers with exactly one request in flight each.
[`scripts/localapi.sh`](scripts/localapi.sh) runs a local pool;
[`modal_localapi.py`](modal_localapi.py) runs the same image on Modal as a
*function* rather than a web endpoint, since a synchronous HTTP proxy times out
around 150 s and evaluations take up to 40 minutes.

---

## Reproducing

```bash
./scripts/reproduce.sh check       # offline: tests + an optimiser run on the built-in surrogate
./scripts/reproduce.sh evaluator   # start 10 local API containers (needs Docker)
./scripts/reproduce.sh diagnose    # per-game fidelity signal-to-noise
./scripts/reproduce.sh search      # medium-direct search, one loop per game
./scripts/reproduce.sh confirm     # race the leaders to n>=3
./scripts/reproduce.sh bundle      # print the submission JSON
```

`check` needs only Python and finishes in about a minute — it runs the tests and
a full optimiser loop against [`mock.py`](src/ttbalance/mock.py), an offline
surrogate of the API. Later stages need a real evaluator and take hours.

For Modal: `modal deploy modal_localapi.py`, then `--backend modal`.

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
  transfer.py       per-game medium->full transfer rates
  calibrate.py      medium-estimate -> full-score calibration
  mock.py           offline surrogate, so the pipeline runs with no API
  optimizers/       random, hill, pbil, ea, nori
scripts/            evaluator pool, search loops, confirmation, screening
docs/COMPETITION.md task, API, scoring
```

44 tests: `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

---

## What did not work

Negative results, recorded because they carry most of the information.

* **Ranking by `fast`.** Hours of search walking the wrong way for Exploding
  Kittens. The fix was per-game diagnosis, not a better optimiser.
* **Bundling several games per submission.** Transfer collapsed to 0.13 because
  one component rested on an n=1 estimate.
* **Trusting the component diagnostic at `fast`.** Twice produced a convincing
  direction (`SEETHEFUTURE=6`, `NOPE=10`, each roughly halving matrix distance)
  that `medium` then rejected. The diagnostic is sound; the fidelity it was read
  at was not.
* **A variance heuristic** — that well-balanced configurations should vary more
  across repeats, since their measured distance is mostly noise. Tested against
  two configurations with known full scores; refuted (sd 15.3 vs 16.3, the wrong
  way round).

## Acknowledgements

The [TAG framework](https://github.com/GAIGResearch/TabletopGames) and the
competition organisers, for shipping a local evaluator image — the whole
approach depends on running evaluations off the hosted queue.
