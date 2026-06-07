"""
BN threshold calibration — analyze bn_training_{asset}.jsonl to find
the bn_d3s threshold at entry that reliably predicts a 10c+ winner move.

Usage:
    python3 analyze_bn_thresholds.py --asset xrp
    python3 analyze_bn_thresholds.py --asset btc
    python3 analyze_bn_thresholds.py --asset xrp --target 0.10
"""
import json, sys, argparse, collections
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--asset', default='xrp')
parser.add_argument('--target', type=float, default=0.10,
                    help='Winner profit target in dollars (default 0.10 = 10c)')
args = parser.parse_args()

FILE = Path(f'bn_training_{args.asset}.jsonl')
if not FILE.exists():
    print(f'ERROR: {FILE} not found. Run from the bot directory.')
    sys.exit(1)

TARGET = args.target

# Load all events
records = []
with open(FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

print(f'\nLoaded {len(records)} records from {FILE}')

# Pair dm_entry with dm_exit by trade_id
entries = {}   # trade_id -> entry event
exits   = {}   # trade_id -> exit event

for r in records:
    ev   = r.get('event', {})
    etype = ev.get('type')
    tid   = ev.get('trade_id')
    if not tid:
        continue
    if etype == 'dm_entry':
        entries[tid] = ev
    elif etype == 'dm_exit':
        exits[tid] = ev

print(f'DM entries: {len(entries)}   DM exits: {len(exits)}')

paired = []
for tid, entry in entries.items():
    if tid not in exits:
        continue
    ex = exits[tid]
    winner_side  = entry.get('winner')
    clob_winner  = entry.get('clob_winner', 0)
    clob_loser   = entry.get('clob_loser', 0)
    bn_d3s_entry = entry.get('d3s')          # BN 3s delta at entry (fraction, e.g. -0.000278)
    exit_avg     = ex.get('avg')
    exit_side    = ex.get('side')
    pnl          = ex.get('pnl')

    if bn_d3s_entry is None or clob_winner is None or exit_avg is None:
        continue

    # Winner profit = exit price - entry price (if exit side == winner side)
    if exit_side == winner_side:
        winner_profit = round(exit_avg - clob_winner, 4)
    else:
        winner_profit = None   # exited loser — complex case, skip for threshold calc

    reached_target = (winner_profit is not None and winner_profit >= TARGET)

    paired.append({
        'tid':           tid,
        'winner':        winner_side,
        'clob_winner':   clob_winner,
        'clob_loser':    clob_loser,
        'bn_d3s':        bn_d3s_entry,     # raw fraction
        'bn_d3s_pct':    round(bn_d3s_entry * 100, 6) if bn_d3s_entry else None,
        'winner_profit': winner_profit,
        'pnl':           pnl,
        'reached_target': reached_target,
    })

print(f'Paired trades: {len(paired)}')
if not paired:
    print('\nNot enough paired trades to calibrate. Run more markets.')
    sys.exit(0)

# ── Summary stats ──────────────────────────────────────────────────────────────
profitable = [p for p in paired if p['winner_profit'] is not None and p['winner_profit'] >= TARGET]
losers     = [p for p in paired if p['winner_profit'] is not None and p['winner_profit'] < TARGET]
blind      = [p for p in paired if p['winner_profit'] is None]

print(f'\n── Trade outcomes ────────────────────────────────────────────────')
print(f'  Reached {TARGET*100:.0f}c target: {len(profitable)} / {len(paired)}')
print(f'  Missed target:          {len(losers)} / {len(paired)}')
print(f'  Exited loser (blind):   {len(blind)}')

if len(paired) < 5:
    print('\nNeed at least 5 paired trades for threshold analysis.')
    sys.exit(0)

# ── BN d3s distribution at entry ──────────────────────────────────────────────
print(f'\n── BN d3s at entry (%) — profitable vs missed ────────────────────')

def pct_stats(vals):
    if not vals:
        return 'n/a'
    s = sorted(vals)
    n = len(s)
    return (f'min={s[0]*100:.4f}%  p25={s[n//4]*100:.4f}%  '
            f'median={s[n//2]*100:.4f}%  p75={s[3*n//4]*100:.4f}%  '
            f'max={s[-1]*100:.4f}%  n={n}')

prof_d3s = [p['bn_d3s'] for p in profitable if p['bn_d3s'] is not None]
miss_d3s = [p['bn_d3s'] for p in losers     if p['bn_d3s'] is not None]

print(f'  Profitable: {pct_stats(prof_d3s)}')
print(f'  Missed:     {pct_stats(miss_d3s)}')

# ── Threshold sweep — precision / recall at each |bn_d3s| threshold ───────────
print(f'\n── Threshold sweep — |bn_d3s| >= threshold → predict profitable ──')
print(f'  {"Threshold%":>12}  {"Precision":>10}  {"Recall":>8}  {"Trades":>7}  {"Blocked":>8}')

thresholds = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
best_f1 = 0
best_thresh = None

for t in thresholds:
    t_frac = t / 100
    # Trades where |bn_d3s| >= threshold
    entered = [p for p in paired if p['bn_d3s'] is not None and abs(p['bn_d3s']) >= t_frac]
    blocked = len(paired) - len(entered)
    if not entered:
        continue
    tp = sum(1 for p in entered if p['reached_target'])
    fp = len(entered) - tp
    fn = sum(1 for p in profitable if p['bn_d3s'] is not None and abs(p['bn_d3s']) < t_frac)
    precision = tp / len(entered) if entered else 0
    recall    = tp / len(profitable) if profitable else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f'  {t:>11.3f}%  {precision:>10.2%}  {recall:>8.2%}  {len(entered):>7}  {blocked:>8}')
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f'\n── Recommendation ────────────────────────────────────────────────')
if best_thresh:
    print(f'  Best threshold (F1={best_f1:.2f}): |bn_d3s| >= {best_thresh:.3f}%')
    print(f'  Set in code as: _DM_BN_CONFIRM_THRESH = {best_thresh/100:.5f}')
    print(f'  (replace the current BN_DM_ENTRY_TOLERANCE bypass logic)')
else:
    print('  Insufficient data for recommendation — collect more trades.')

print(f'\n── Per-trade detail ──────────────────────────────────────────────')
print(f'  {"trade_id":>20}  {"winner":>6}  {"bn_d3s%":>10}  {"clob_win":>9}  {"profit":>8}  {"hit":>5}')
for p in sorted(paired, key=lambda x: x['bn_d3s'] or 0):
    d = f"{p['bn_d3s']*100:.4f}%" if p['bn_d3s'] is not None else '?'
    pr = f"{p['winner_profit']:+.4f}" if p['winner_profit'] is not None else 'n/a'
    hit = '✅' if p['reached_target'] else '❌'
    print(f"  {p['tid']:>20}  {p['winner']:>6}  {d:>10}  "
          f"{p['clob_winner']:>9.3f}  {pr:>8}  {hit:>5}")
