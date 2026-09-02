# Submission ledger

Every entry submitted, with the score the leaderboard gave it. Best: **3652.5**.

| entry | full score | what changed vs the previous best |
|---|---|---|
| `probe-w7-lo` | **3652.5** | 7 Wonders → a low-`medium` configuration |
| `probe-w7-hi` | 3648.3 | 7 Wonders → the highest-`medium` configuration |
| `probe-ek-895` | 3642.4 | Exploding Kittens → 895.4 (n=7) |
| `probe-w7-revert` | 3637.8 | 7 Wonders reverted to the v6 configuration (+30.6) |
| `probe-cs-revert` | 3636.3 | Can't Stop reverted to v6 (−1.5, so kept the newer one) |
| `sub-B-dom-w7div` | 3627.7 | Dominion swap + a maximally different 7 Wonders |
| `sub-A-dominion` | 3620.1 | Dominion → higher `fast` configuration (**−32.4**) |
| `medium-direct-v7` | 3607.2 | three games at once: +45 measured, +1.6 real |
| `medium-direct-v6` | 3605.6 | Can't Stop only |
| `medium-direct-v5` | 3562.3 | all four games, one on a single measurement |
| `medium-direct-v4` | 3553.3 | Exploding Kittens only |
| `medium-direct-v3` | 3513.7 | Exploding Kittens only |
| `medium-confirmed-v2` | 3493.1 | first bundle confirmed at 3+ repeats |
| `confirmed-n7-v1` | 3490.0 | first submission |

## What the ledger shows

**Single-game changes carry over; bundles do not.**

| submission | games changed | measured gain | real gain | ratio |
|---|---|---|---|---|
| v2 → v3 | 1 | +24.2 | +21.0 | 0.87 |
| v3 → v4 | 1 | +13.4 | +39.0 | 2.91 |
| v4 → v5 | 4 | +74.4 | +9.3 | **0.13** |
| v6 → v7 | 3 | +44.9 | +1.6 | **0.04** |
| v7 → `probe-w7-revert` | 1 | −14.8 | **+30.6** | — |

The last row is the sharpest: our own measurement said that change made things
*worse* by 14.8, and it was worth +30.6. At `medium`, 7 Wonders configurations
are separated by less than the noise between repeats (see §2 of the README), so
the measurement simply could not tell them apart.
