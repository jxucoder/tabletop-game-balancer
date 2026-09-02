# Competition reference

Everything below is transcribed from the official competition site
(<https://balance-competition.tabletopgames.ai>) and the organisers' example
repo, gathered 2026-08-19. Re-check before the deadline — the site is the
authority, not this file.

## Dates

| Milestone | Date |
|---|---|
| Competition opened | 28 Feb 2026 |
| Testing period ended | 31 Mar 2026 |
| **Submissions close** | **25 Aug 2026** |
| Winner announced (CoG 2026) | 3 Sep 2026 |

## The task

Black-box optimisation. You choose a rule configuration for a game (e.g.
`HAND_SIZE=5`), the server plays that configuration many times in the TAG
framework using four fixed agents, and returns a **balance score**. Maximise it.

Agents used in evaluation: **Elite** (tuned MCTS), **Good** (MCTS at half the
time budget), **OSLA** (one-step-lookahead greedy), **Random**.

## Scoring

Per game, maximum 1000; four games, so the leaderboard maxes at **4000**.

- Dominion, ExplodingKittens, CantStop:
  `score = 1000 - scaling * (matchup distance - win-rate distance)`,
  with a target first-player win rate of 50%.
- Wonders7: `score = 1000 - scaling * winrate distance`.
  No first-player term — turns are simultaneous.

Each game has its own scaling factor so the four contribute equally.

## Run types

| `run_type` | Matchups | Use |
|---|---|---|
| `fast` | 36 | Search. Very noisy. |
| `medium` | 360 | Shortlisting. |
| `full` | 3600 | Final confirmation. |

## Hosted API

Base: `https://balance-competition.tabletopgames.ai/api/`. Sign in with GitHub,
generate a key under settings. Asynchronous — three calls per evaluation.

```
POST /submit_run       {"game","params","api_key","run_type"}  -> {"runID": int}
GET  /query_run?id=N                                           -> {"run_status": "running|results|complete"}
GET  /retrieve_result?id=N&api_key=KEY                         -> {"score": float, "runID": int}
```

Poll `query_run` about every 10s until `complete`, then `retrieve_result`.
`query_run` needs no API key. 400 = bad params, 401 = bad key.

## Local API (recommended for search)

The organisers publish a Docker image with no queue and one synchronous call.
This is where you should spend your search budget.

```bash
docker run --rm -p 3000:3000 longhousedev/localapi
```

```
POST http://localhost:3000/api/run_game
     {"game","params","run_type","timeout"(optional ms)} -> {"score": float} | {"error": ...}
```

Note it is `run_game` locally versus `submit_run` on the hosted server, and
there is no `api_key` field.

## Submitting an entry

The API is for evaluation only. Final parameters go through the **Submit Entry**
form on the website, which is what places you on the leaderboard.

## Parameter space

Canonical definition lives in [`config/valid_params.json`](../config/valid_params.json),
taken from the organisers' `balance-comp-examples` repo.

| Game | Params | Notes |
|---|---|---|
| Dominion | 10 | `CARDS` selects exactly 10 of 26 kingdom cards |
| ExplodingKittens | 14 | one boolean (`nopeOwnCards`), rest small integers |
| Wonders7 | 29 | `wonders` selects 4–7 of the 7 wonders |
| CantStop | 13 | column maxima, `COLUMNS_TO_WIN`, `MARKERS` |

**Known discrepancy:** the website's Dominion table spells the pile-exhaustion
parameter `PILES_EXHAUSED_FOR_GAME_END` (missing "T"), while
`valid_params.json` uses `PILES_EXHAUSTED_FOR_GAME_END`. This repo follows the
JSON. Run `ttbalance probe` against a live API to confirm which one is accepted
before spending a large budget.

## Links

- Competition home: <https://balance-competition.tabletopgames.ai>
- CoG 2026 competitions: <https://cog2026.org/competitions>
- Organisers' examples: <https://github.com/longhousedev/balance-comp-examples>
- TAG framework: <https://github.com/GAIGResearch/TabletopGames>
