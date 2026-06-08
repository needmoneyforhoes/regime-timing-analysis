# regime-timing-analysis

Offline analysis scripts for Polymarket 5-minute crypto up/down markets: regime persistence, BN-threshold calibration, and `dn_live` gate discrimination. Split out of the main trading monorepo.

Read-only. The scripts ingest recorded market history and emit reports. They do not trade or touch a wallet.

## Scripts

- `regime_tick_analyzer_v2.py`: tests whether cumulative regime features at fire time (cd around 150) predict end-of-market regime. Adds late-flip detectability (final 120s) and barbell-loss disaster signatures (loss >= $5). Uses the `tick_columns`/`ticks` schema.
- `regime_tick_analyzer.py`: v1 of the above. Kept for reference; predates the corrected tick schema. Superseded by v2.
- `analyze_bn_thresholds.py`: pairs `dm_entry`/`dm_exit` by trade_id and sweeps `|bn_d3s|` entry thresholds for precision/recall/F1 against a winner move target.
- `dnlive_analysis.py`: re-simulates the `dn_live` gate from raw ticks and searches for a leak-free discriminator between correct and wrong blocks. Fisher/t-test, Bonferroni correction, walk-forward OOS split, 10k-iteration permutation test.
- `post_fix_analysis.py`: compares pre/post WS-lag-fix PnL windows, allowlist auto-demote evolution, and trending-market loss patterns. 4h cutoff hardcoded.

## Requirements

Python 3.8+. `dnlive_analysis.py` needs `numpy` and `scipy`; the rest are stdlib.

```bash
pip install numpy scipy
```

## Usage

Run from a directory holding the data files.

```bash
python3 regime_tick_analyzer_v2.py market_history.jsonl --limit 50
python3 analyze_bn_thresholds.py --asset xrp --target 0.10
python3 dnlive_analysis.py
python3 post_fix_analysis.py
```

`dnlive_analysis.py` and `post_fix_analysis.py` use hardcoded paths under `$DATA_DIR` (`dnlive_analysis.py` also probes `./market_history.jsonl`). Outputs land next to the data: `regime_tick_report.txt`, `regime_persistence_data.json`, `dnlive_analysis_results.json`, `dnlive_analysis_summary.txt`.

## Data

Data is not committed (`.gitignore` excludes `*.jsonl`/`*.json`/`*.pkl`/`*.csv`). Provision from the private `polymarket-data` repo into the working dir (`$DATA_DIR` for the two hardcoded scripts):

- `market_history.jsonl`: per-market tick/event records. All analyzers.
- `bn_training_{btc,xrp}.jsonl`: paired DM entry/exit events. `analyze_bn_thresholds.py`.
- `market_recap_history.jsonl`, `live_allowlist.json`: recap and allowlist evolution. `dnlive_analysis.py`, `post_fix_analysis.py`.
