"""
dn_live gate deep analysis — run on VPS where market_history.jsonl is available.

Purpose:
  Simulate dn_live from scratch on ALL markets using raw tick data.
  Find a statistically valid discriminator that separates UP-win (correct block)
  from DN-win (wrong block) markets — available at trigger time, no future bias.

Statistical methodology:
  - Fisher exact / t-test per candidate feature
  - Bonferroni correction for multiple tests
  - Walk-forward OOS split (first 60% IS, last 40% OOS)
  - Permutation test (10k iterations)
  - Autocorrelation check on outcomes

Run:
  cd .
  python3 dnlive_analysis.py

Outputs:
  dnlive_analysis_results.json   — full per-market results
  dnlive_analysis_summary.txt    — human-readable report
"""

import json, os, sys, numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────
MARKET_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH   = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')
OUTPUT_JSON          = 'dnlive_analysis_results.json'
OUTPUT_TXT           = 'dnlive_analysis_summary.txt'

# dn_live trigger parameters (current live values)
DNLIVE_UP_THRESHOLD  = 0.50   # UP ask >= this
DNLIVE_CONSEC        = 2      # consecutive 10s samples
DNLIVE_MAX_CD        = 210    # only check at cd <= this

N_PERMUTATIONS       = 10_000
IS_SPLIT             = 0.60   # first 60% = in-sample, last 40% = OOS
BONFERRONI_N_TESTS   = 6      # number of features we test
ALPHA                = 0.05
ALPHA_BONF           = ALPHA / BONFERRONI_N_TESTS

# ── HELPERS ────────────────────────────────────────────────────────────────
def pnl(fires): return round(sum(f.get('hypo_pnl', 0) for f in fires), 4)

def simulate_dnlive(ticks, cols):
    """
    Simulate dn_live trigger from raw tick data.
    Returns (trig_cd, features_at_trigger) or (None, None) if never triggers.

    dn_live logic (from bot code):
      Every ~10s: if up_ask >= 0.50 → consec += 1, else consec = 0
      If consec >= 2 AND cd <= 210 → TRIGGERED
    """
    ci = {c: i for i, c in enumerate(cols)}
    cd_i  = ci.get('cd', 0)
    ua_i  = ci.get('up_ask', 2)
    da_i  = ci.get('dn_ask', 4)
    bn_i  = ci.get('bn_delta_pct', 12)
    cr_i  = ci.get('crowd_conviction', 20)  # 0 if absent
    cs_i  = ci.get('crowd_side', 19)
    ue_i  = ci.get('up_ema', 13)
    de_i  = ci.get('dn_ema', 14)
    dd_i  = ci.get('dn_depth', 6)

    consec     = 0
    last_10s   = None   # timestamp approximation via cd
    prev_cd    = 999

    # We simulate 10s sampling by taking one sample per ~10cd drop
    last_sample_cd = 9999

    for t in ticks:
        cd = t[cd_i]
        if cd is None or cd <= 0:
            continue
        up_ask = t[ua_i] or 0
        dn_ask = t[da_i] or 0

        # Sample every ~10s (cd drops ~10 per 10 real seconds)
        if last_sample_cd - cd < 9.0:
            continue
        last_sample_cd = cd

        if cd > DNLIVE_MAX_CD:
            # still accumulating
            if up_ask >= DNLIVE_UP_THRESHOLD:
                consec += 1
            else:
                consec = 0
            continue

        # cd <= 210: check gate
        if up_ask >= DNLIVE_UP_THRESHOLD:
            consec += 1
        else:
            consec = 0

        if consec >= DNLIVE_CONSEC:
            # TRIGGERED — collect features at this tick
            features = {
                'trig_cd':      round(cd, 1),
                'up_ask':       round(up_ask, 3),
                'dn_ask':       round(dn_ask, 3),
                'spread':       round(dn_ask - up_ask, 3),
                'bn_delta':     round(t[bn_i] or 0, 5) if bn_i < len(t) else None,
                'crowd_conv':   round(t[cr_i] or 0, 3) if cr_i < len(t) else 0,
                'crowd_side':   t[cs_i] if cs_i < len(t) else None,
                'up_ema':       round(t[ue_i] or 0, 4) if ue_i < len(t) else None,
                'dn_ema':       round(t[de_i] or 0, 4) if de_i < len(t) else None,
                'dn_depth':     round(t[dd_i] or 0, 1) if dd_i < len(t) else None,
            }
            return cd, features

    return None, None


def get_entry_features(ticks, cols):
    """Features from the entry window (cd=240-310), no future bias."""
    ci = {c: i for i, c in enumerate(cols)}
    cd_i = ci.get('cd', 0)
    ua_i = ci.get('up_ask', 2)
    da_i = ci.get('dn_ask', 4)
    bn_i = ci.get('bn_delta_pct', 12)
    dd_i = ci.get('dn_depth', 6)

    entry_spreads, entry_depths, entry_bns = [], [], []
    for t in ticks:
        cd = t[cd_i]
        if cd is None: continue
        if 240 <= cd <= 310:
            ua = t[ua_i] or 0; da = t[da_i] or 0
            entry_spreads.append(da - ua)
            if dd_i < len(t) and t[dd_i]: entry_depths.append(t[dd_i])
            if bn_i < len(t) and t[bn_i] is not None: entry_bns.append(t[bn_i])

    return {
        'entry_spread_avg': round(np.mean(entry_spreads), 4) if entry_spreads else None,
        'entry_depth_avg':  round(np.mean(entry_depths), 1)  if entry_depths  else None,
        'entry_bn_avg':     round(np.mean(entry_bns), 5)     if entry_bns     else None,
    }


# ── LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading market history...", flush=True)
if not Path(MARKET_HISTORY_PATH).exists():
    # Try alternative paths
    for alt in ['~/polymarket-bot/market_history.jsonl',
                '~/market_history.jsonl',
                './market_history.jsonl']:
        p = Path(os.path.expanduser(alt))
        if p.exists():
            MARKET_HISTORY_PATH = str(p)
            break
    else:
        print(f"ERROR: market_history.jsonl not found. Tried:")
        print(f"  {MARKET_HISTORY_PATH}")
        print("Please provide the correct path.")
        sys.exit(1)

# Load market history (tick data)
mkt_hist = {}
n_loaded = 0
with open(MARKET_HISTORY_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            slug = d.get('slug')
            if slug and d.get('ticks') and d.get('tick_columns'):
                mkt_hist[slug] = d
                n_loaded += 1
        except Exception:
            pass
print(f"  Loaded {n_loaded} markets with tick data")

# Load recap history (fire outcomes)
recap = {}
n_recap = 0
if Path(RECAP_HISTORY_PATH).exists():
    with open(RECAP_HISTORY_PATH) as f:
        for line in f:
            try:
                d = json.loads(line)
                slug = d.get('slug')
                if slug and d.get('fires') and d.get('winner') in ('UP','DN'):
                    recap[slug] = d
                    n_recap += 1
            except Exception:
                pass
print(f"  Loaded {n_recap} markets with fire data")

# ── SIMULATE dn_live ON ALL MARKETS ────────────────────────────────────────
print("\nSimulating dn_live on all markets...", flush=True)

results = []
n_triggered = 0

# Markets that appear in both history files
common_slugs = sorted(set(mkt_hist.keys()) & set(recap.keys()))
print(f"  Markets in both files: {len(common_slugs)}")

for slug in common_slugs:
    mkt    = mkt_hist[slug]
    rec    = recap[slug]
    ticks  = mkt['ticks']
    cols   = mkt['tick_columns']
    winner = rec['winner']
    fires  = rec['fires']

    trig_cd, trig_feats = simulate_dnlive(ticks, cols)
    entry_feats = get_entry_features(ticks, cols)

    # Fires that would be blocked (DN fires after trig_cd)
    # In the bot: once dn_live triggers, all subsequent DN fires get WBLOCK
    blocked_dn = []
    if trig_cd is not None:
        for f in fires:
            if (f.get('side') == 'DN' and
                    f.get('cd', 9999) < trig_cd and  # fire cd < trig_cd (lower cd = later)
                    not f.get('pre_gate_held') and
                    not f.get('opp_gate_blocked')):
                blocked_dn.append(f)

    if trig_cd is not None:
        n_triggered += 1

    results.append({
        'slug':         slug,
        'winner':       winner,
        'triggered':    trig_cd is not None,
        'trig_cd':      trig_cd,
        'trig_feats':   trig_feats,
        'entry_feats':  entry_feats,
        'n_blocked_dn': len(blocked_dn),
        'pnl_blocked_dn': pnl(blocked_dn),
        # s0 = current hypo_gc (including current gate state in recap)
        # For simulation we use what's in recap as baseline
        's0': sum(f.get('hypo_pnl', 0) for f in fires
                  if not f.get('pre_gate_held') and not f.get('opp_gate_held')
                  and not f.get('opp_gate_would_block') and not f.get('opp_gate_blocked')),
    })

print(f"  dn_live triggered in {n_triggered}/{len(results)} markets "
      f"({n_triggered/len(results)*100:.1f}%)")

# ── ANALYSIS ───────────────────────────────────────────────────────────────
trig_markets = [r for r in results if r['triggered']]
up_trig  = [r for r in trig_markets if r['winner'] == 'UP']   # correct blocks
dn_trig  = [r for r in trig_markets if r['winner'] == 'DN']   # wrong blocks

print(f"\n  Triggered markets: {len(trig_markets)}")
print(f"    UP win (correct, gate helps): {len(up_trig)}")
print(f"    DN win (wrong, gate hurts):   {len(dn_trig)}")
if trig_markets:
    prec = len(up_trig) / len(trig_markets)
    print(f"    Precision: {prec*100:.1f}%")

# PnL impact
correct_pnl = sum(r['pnl_blocked_dn'] for r in up_trig)
wrong_pnl   = sum(r['pnl_blocked_dn'] for r in dn_trig)
net_gate_ev = correct_pnl + wrong_pnl  # positive = removal gains this much

print(f"\n  PnL of simulated DN blocks:")
print(f"    Correct (UP win): {correct_pnl:+.2f} (losses avoided by blocking)")
print(f"    Wrong   (DN win): {wrong_pnl:+.2f}  (profits missed by blocking)")
print(f"    Net (removal gain): {net_gate_ev:+.2f}  "
      f"({'removing helps' if net_gate_ev > 0 else 'keeping helps'})")
print(f"    Per triggered market: {net_gate_ev/len(trig_markets) if trig_markets else 0:+.3f}")

# ── FEATURE ANALYSIS: find discriminator ──────────────────────────────────
print(f"\n{'═'*70}")
print(f"FEATURE ANALYSIS (Bonferroni α={ALPHA_BONF:.4f} = {ALPHA}/{BONFERRONI_N_TESTS} tests)")
print(f"{'═'*70}")
print(f"UP win (correct, n={len(up_trig)}) vs DN win (wrong, n={len(dn_trig)})")
print()

candidates = [
    ('trig_cd',        lambda r: r['trig_cd'],                         'trigger cd (higher=earlier)'),
    ('bn_delta',       lambda r: r['trig_feats']['bn_delta'] if r['trig_feats'] else None, 'BN delta at trigger'),
    ('spread',         lambda r: r['trig_feats']['spread'] if r['trig_feats'] else None,   'dn-up spread at trigger'),
    ('crowd_conv',     lambda r: r['trig_feats']['crowd_conv'] if r['trig_feats'] else 0,  'crowd conviction at trigger'),
    ('entry_spread',   lambda r: r['entry_feats']['entry_spread_avg'] if r['entry_feats'] else None, 'entry spread (cd=240-310)'),
    ('entry_depth',    lambda r: r['entry_feats']['entry_depth_avg'] if r['entry_feats'] else None,  'entry DN depth avg'),
]

feature_results = []
for feat_name, feat_fn, feat_label in candidates:
    c_vals = [feat_fn(r) for r in up_trig  if feat_fn(r) is not None]
    w_vals = [feat_fn(r) for r in dn_trig  if feat_fn(r) is not None]
    if len(c_vals) < 2 or len(w_vals) < 2:
        print(f"  {feat_name:15}: SKIP (insufficient data c={len(c_vals)}, w={len(w_vals)})")
        continue

    c_arr, w_arr = np.array(c_vals), np.array(w_vals)
    t_stat, p_raw = stats.ttest_ind(c_arr, w_arr, equal_var=False)
    p_bonf = min(p_raw * BONFERRONI_N_TESTS, 1.0)
    sig = "*** SIGNIFICANT ***" if p_bonf < ALPHA else "ns"
    mean_c, mean_w = np.mean(c_arr), np.mean(w_arr)

    print(f"  {feat_name:16}: correct_mean={mean_c:>+8.4f}  wrong_mean={mean_w:>+8.4f}  "
          f"Δ={mean_c-mean_w:>+8.4f}  p_raw={p_raw:.4f}  p_bonf={p_bonf:.4f}  {sig}")

    feature_results.append({
        'name': feat_name, 'label': feat_label,
        'c_vals': c_vals, 'w_vals': w_vals,
        'mean_c': mean_c, 'mean_w': mean_w,
        'p_raw': p_raw, 'p_bonf': p_bonf,
        'significant': p_bonf < ALPHA,
    })

# ── BEST DISCRIMINATOR ─────────────────────────────────────────────────────
sig_feats = [f for f in feature_results if f['significant']]
best_feat  = min(feature_results, key=lambda f: f['p_raw']) if feature_results else None

print()
if sig_feats:
    print(f"✅ SIGNIFICANT discriminators found (Bonferroni p<{ALPHA}):")
    for f in sig_feats:
        print(f"   {f['name']}: correct={f['mean_c']:+.4f} vs wrong={f['mean_w']:+.4f} (p_bonf={f['p_bonf']:.4f})")
else:
    print(f"❌ No Bonferroni-significant discriminator found with n={len(trig_markets)} trigger markets.")
    if best_feat:
        print(f"   Best candidate: {best_feat['name']} (p_bonf={best_feat['p_bonf']:.4f}, need p<{ALPHA_BONF:.4f})")
        n_needed = int(np.ceil(
            (stats.norm.ppf(1-ALPHA_BONF/2) + stats.norm.ppf(0.80))**2 *
            (np.std(best_feat['c_vals'])**2/len(best_feat['c_vals']) +
             np.std(best_feat['w_vals'])**2/len(best_feat['w_vals'])) /
            (best_feat['mean_c'] - best_feat['mean_w'])**2 * len(trig_markets)
        ))
        print(f"   Estimated trigger markets needed for 80% power: ~{n_needed}")

# ── THRESHOLD SEARCH (if best discriminator exists) ────────────────────────
if best_feat and best_feat['p_raw'] < 0.20:
    print()
    print(f"{'─'*60}")
    print(f"THRESHOLD SEARCH: {best_feat['name']}")
    print(f"{'─'*60}")
    all_vals = best_feat['c_vals'] + best_feat['w_vals']
    labels   = [1]*len(best_feat['c_vals']) + [0]*len(best_feat['w_vals'])
    percentiles = [10,20,30,40,50,60,70,80,90]
    print(f"  {'Threshold':>10} {'Prec@above':>10} {'n_kept':>7} {'EV_impact':>10} {'EV_vs_nogate':>13}")
    for pct in percentiles:
        thresh = np.percentile(all_vals, pct)
        # If we only apply dn_live when feature > thresh (higher = more UP-dominant)
        # or < thresh depending on direction
        direction = 1 if best_feat['mean_c'] > best_feat['mean_w'] else -1
        above_correct = sum(1 for v,l in zip(all_vals,labels) if (v*direction) >= (thresh*direction) and l==1)
        above_wrong   = sum(1 for v,l in zip(all_vals,labels) if (v*direction) >= (thresh*direction) and l==0)
        kept = above_correct + above_wrong
        prec = above_correct / kept if kept > 0 else 0
        # EV: saves |correct_pnl| per correct, misses |wrong_pnl| per wrong
        # Simplified: compare precision to baseline
        baseline_prec = len(up_trig) / len(trig_markets)
        ev_change = (prec - baseline_prec) * abs(correct_pnl / len(up_trig) if up_trig else 1)
        print(f"  {thresh:>10.4f} {prec*100:>9.1f}% {kept:>7} {ev_change:>+10.3f} {'see below':>13}")

# ── PERMUTATION TEST ON REMOVAL EV ─────────────────────────────────────────
print()
print(f"{'─'*60}")
print(f"PERMUTATION TEST: is removing dn_live EV-positive?")
print(f"{'─'*60}")
deltas = [r['pnl_blocked_dn'] for r in results]  # 0 if not triggered
obs    = np.mean(deltas)
np.random.seed(42)
perm_dist = [np.mean(np.random.choice([-1,1], size=len(deltas)) * deltas)
             for _ in range(N_PERMUTATIONS)]
p_perm = np.mean(np.abs(perm_dist) >= abs(obs))
print(f"  Observed mean delta (removal): {obs:+.4f}/mkt")
print(f"  Permutation p: {p_perm:.4f} {'*' if p_perm<ALPHA else 'ns'}")

# ── WALK-FORWARD SPLIT ─────────────────────────────────────────────────────
print()
print(f"{'─'*60}")
print(f"WALK-FORWARD SPLIT (IS: first {IS_SPLIT*100:.0f}%  OOS: last {(1-IS_SPLIT)*100:.0f}%)")
print(f"{'─'*60}")
split_idx = int(len(results) * IS_SPLIT)
is_res  = results[:split_idx]
oos_res = results[split_idx:]
for label, subset in [('IS ', is_res), ('OOS', oos_res)]:
    t_sub = [r for r in subset if r['triggered']]
    up_t  = [r for r in t_sub if r['winner']=='UP']
    dn_t  = [r for r in t_sub if r['winner']=='DN']
    net   = sum(r['pnl_blocked_dn'] for r in t_sub)
    prec  = len(up_t)/len(t_sub)*100 if t_sub else 0
    mean_d= np.mean([r['pnl_blocked_dn'] for r in subset])
    print(f"  {label}: n={len(subset):3}  triggers={len(t_sub):2}  "
          f"prec={prec:5.1f}%  net_removal_gain={net:>+8.2f}  "
          f"delta/mkt={mean_d:>+7.3f}")

# ── AUTOCORRELATION ────────────────────────────────────────────────────────
print()
print(f"{'─'*60}")
print("AUTOCORRELATION CHECK")
print(f"{'─'*60}")
trig_seq = [1 if r['triggered'] else 0 for r in results]
pnl_seq  = [r['pnl_blocked_dn'] for r in results]
if len(pnl_seq) > 2:
    ac1 = np.corrcoef(pnl_seq[:-1], pnl_seq[1:])[0,1]
    ac2 = np.corrcoef(pnl_seq[:-2], pnl_seq[2:])[0,1] if len(pnl_seq)>3 else 0
    print(f"  PnL delta autocorr lag-1: {ac1:.4f}  lag-2: {ac2:.4f}")
    print(f"  (|r|<0.10 = effectively independent = clean signal)")
    trig_ac = np.corrcoef(trig_seq[:-1], trig_seq[1:])[0,1]
    print(f"  Trigger autocorr lag-1:   {trig_ac:.4f}  (clustering?)")

# ── VERDICT & RECOMMENDATION ───────────────────────────────────────────────
print()
print(f"{'═'*70}")
print("QUANTITATIVE VERDICT")
print(f"{'═'*70}")
print(f"  n_total = {len(results)}  n_triggered = {n_triggered}  "
      f"precision = {len(up_trig)/len(trig_markets)*100:.1f}%")
print(f"  Net removal EV: {net_gate_ev:+.2f}  per triggered mkt: "
      f"{net_gate_ev/len(trig_markets) if trig_markets else 0:+.3f}")
print(f"  Permutation p: {p_perm:.4f}")
print()

if p_perm < ALPHA and net_gate_ev > 0:
    print("  → REMOVE dn_live: significantly EV-positive to disable")
elif p_perm < ALPHA and net_gate_ev < 0:
    print("  → KEEP dn_live: significantly EV-positive to keep")
elif sig_feats:
    print(f"  → MODIFY dn_live: add filter on {sig_feats[0]['name']}")
    print(f"    (Bonferroni-significant discriminator found)")
else:
    print("  → INCONCLUSIVE: insufficient data for statistically rigorous decision")
    print(f"  → ACTION: keep current gate, accumulate more trigger markets")
    n_trig_needed = max(30, 2 * len(trig_markets))
    print(f"    Target: {n_trig_needed} trigger markets before re-evaluating")

# ── SAVE RESULTS ───────────────────────────────────────────────────────────
print()
print(f"Saving results to {OUTPUT_JSON}...", flush=True)

out = {
    'n_total': len(results),
    'n_triggered': n_triggered,
    'precision': len(up_trig)/len(trig_markets) if trig_markets else 0,
    'correct_pnl': correct_pnl,
    'wrong_pnl': wrong_pnl,
    'net_removal_gain': net_gate_ev,
    'permutation_p': p_perm,
    'feature_results': [
        {k: v for k, v in f.items() if k not in ('c_vals','w_vals')}
        for f in feature_results
    ],
    'trigger_markets': [
        {
            'slug': r['slug'],
            'winner': r['winner'],
            'trig_cd': r['trig_cd'],
            'trig_feats': r['trig_feats'],
            'entry_feats': r['entry_feats'],
            'pnl_blocked_dn': r['pnl_blocked_dn'],
            'n_blocked_dn': r['n_blocked_dn'],
        }
        for r in trig_markets
    ],
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Done. See {OUTPUT_JSON} and run again with results for threshold optimization.")
