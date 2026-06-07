# regime-timing-analysis

Regime-conditional and gate-timing research for Polymarket: tick-level regime
persistence, BN-threshold calibration, and `dn_live` gate discrimination.

These are **offline analysis scripts** split out of the main trading monorepo.
None of them trade or touch a wallet — they ingest recorded market history and
emit reports that inform how the live bot gates and routes 5-minute crypto
up/down strategies.

## Why it exists

The live bot makes mid-market decisions (fire/skip, side routing, gate
triggers) that depend on whether the *current* market regime predicts the
*end-of-market* regime. This repo answers, with proper statistics, whether
those gates have predictive edge at fire time — and calibrates their
thresholds — before anything is wired live.

## What's inside

| Script | Question it answers | Invocation |
| --- | --- | --- |
| `regime_tick_analyzer.py` | At fire time (cd≈150), do cumulative regime features (depth oscillations, lead changes, BN sign-flips, time near 50/50) predict the end-of-market regime? | takes `market_history.jsonl` path |
| `regime_tick_analyzer_v2.py` | Same question, rewritten for the real `tick_columns`/`ticks` schema; adds late-flip detectability (final 120s) and disaster-signature (barbell loss ≥ $5) analysis | takes `market_history.jsonl` path |
| `analyze_bn_thresholds.py` | What `|bn_d3s|` threshold at entry reliably predicts a 10c+ winner move? Pairs `dm_entry`/`dm_exit` and sweeps precision/recall/F1 | `--asset {btc,xrp}` |
| `dnlive_analysis.py` | Re-simulates the `dn_live` gate from raw ticks and hunts a leak-free discriminator separating correct vs wrong blocks. Uses Fisher/t-test, Bonferroni correction, walk-forward OOS split, 10k-iteration permutation test | no args (paths hardcoded) |
| `post_fix_analysis.py` | Did performance change after the WS-lag fix? Compares pre-/post-fix PnL windows, allowlist auto-demote evolution, and trending-market loss patterns | no args (4h cutoff hardcoded) |

`regime_tick_analyzer_v2.py` supersedes v1 — v1 is kept for reference but
predates the corrected tick schema.

## Requirements

- Python 3.8+
- `numpy`, `scipy` (only `dnlive_analysis.py` needs these; the others are
  pure stdlib)

```bash
pip install numpy scipy
```

No wallet, private key, or network access required — these scripts read files
and print/write reports.

## Usage

Run from a directory that contains the recorded data files (see **Data**):

```bash
# Regime persistence — pass the history file explicitly
python3 regime_tick_analyzer_v2.py market_history.jsonl --limit 50

# BN entry-threshold calibration for a given asset
python3 analyze_bn_thresholds.py --asset xrp --target 0.10

# dn_live gate discriminator search (paths are hardcoded to ~/polymarket-bot)
python3 dnlive_analysis.py

# Post-WS-fix performance comparison
python3 post_fix_analysis.py
```

Outputs are written next to the data: `regime_tick_report.txt`,
`regime_persistence_data.json`, `dnlive_analysis_results.json`,
`dnlive_analysis_summary.txt`.

## Data

All scripts read recorded data that is **not** included in this repo (the
`.gitignore` excludes all `*.jsonl`/`*.json`/`*.pkl`/`*.csv`). Provision them
from the private **`polymarket-data`** repo into the working directory (or
`~/polymarket-bot/` for `dnlive_analysis.py`, which expands hardcoded paths):

- `market_history.jsonl` — per-market tick/event records (all analyzers)
- `bn_training_{btc,xrp}.jsonl` — paired DM entry/exit events (`analyze_bn_thresholds.py`)
- `market_recap_history.jsonl`, `live_allowlist.json` — recap + allowlist evolution (`dnlive_analysis.py`, `post_fix_analysis.py`)

> Private research software. No warranty; trades/handles real funds at your own risk.
