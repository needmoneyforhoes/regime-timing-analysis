#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_tick_analyzer.py — Run on VPS against market_history.jsonl
================================================================

Answers the key quant question for regime-conditional deployment:

    "At fire time (e.g. cd=150), do cumulative market features so far
     predict the END-OF-MARKET regime?"

If yes, we can classify regime AT FIRE TIME and route strategies.
If no, regime is unpredictable mid-market and we can't use it as a gate.

Features tracked per market, cumulatively at each tick:
  - depth_oscillations_up_so_far
  - depth_oscillations_dn_so_far
  - lead_changes_so_far
  - time_near_5050_so_far
  - bn_sign_flips_so_far
  - bn_abs_range_so_far (max |bn| seen minus min)

For each market, we snapshot these at multiple fire-time checkpoints
(cd = 200, 150, 100, 50) and compare to END values.

Output: correlations and classification accuracy tables.

USAGE (on VPS):
  python3 regime_tick_analyzer.py /path/to/market_history.jsonl
  (accepts .jsonl or streaming output; reads one market per line)

Output files:
  regime_tick_report.txt        — human-readable diagnostics
  regime_persistence_data.json  — raw numbers for downstream analysis

THIS SCRIPT DOES NOT TRADE. It is pure analysis. Safe to run anytime.
"""

import json
import sys
import argparse
import math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
#  TICK PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def extract_events_and_ticks(market):
    """Pull the events and tick-level series from a market record."""
    events = market.get("events") or []
    ticks = market.get("tick_columns") or []
    # ticks may be list-of-dicts or dict-of-lists depending on bot version
    return events, ticks

def compute_cumulative_features(market):
    """
    Scan through the tick stream and compute cumulative regime features
    at each tick. Returns a list of (cd, features_dict) sorted by cd descending
    (since markets count DOWN from 300 to 0).

    Features:
      depth_osc_up:    count of up-depth sign changes so far
      depth_osc_dn:    count of dn-depth sign changes so far
      lead_changes:    count of times which side was "leading" flipped
      time_near_5050:  cumulative seconds where both sides' prices were near $0.50
      bn_flips:        count of bn_delta sign changes so far
      bn_max_abs:      max absolute bn_delta observed so far
      bn_min_abs:      min absolute bn_delta observed so far (for range)
      bn_flipped_sign: True if bn_delta changed sign at least once
    """
    ticks = market.get("tick_columns") or []
    if not ticks:
        # Some versions store differently; try 'tick_columns' as dict
        return []
    
    # Extract arrays — handle both formats
    if isinstance(ticks, list) and ticks and isinstance(ticks[0], dict):
        # List of tick dicts
        tick_list = sorted(ticks, key=lambda t: t.get("cd", 0), reverse=True)
    elif isinstance(ticks, dict):
        # Dict of parallel arrays
        cds = ticks.get("cd", [])
        tick_list = []
        keys = list(ticks.keys())
        n = len(cds)
        for i in range(n):
            tick_list.append({k: ticks[k][i] if i < len(ticks[k]) else None for k in keys})
        tick_list.sort(key=lambda t: t.get("cd", 0), reverse=True)
    else:
        return []
    
    if not tick_list:
        return []
    
    # Running state
    state = dict(
        depth_osc_up=0, depth_osc_dn=0,
        lead_changes=0, time_near_5050=0,
        bn_flips=0, bn_max_abs=0.0, bn_min_abs=1.0,
        bn_flipped_sign=False,
        prev_up_depth=None, prev_dn_depth=None,
        prev_up_depth_dir=None, prev_dn_depth_dir=None,
        prev_lead=None,
        prev_bn=None, first_bn_sign=None,
        prev_cd=None,
    )
    
    snapshots = []
    
    for t in tick_list:
        cd = t.get("cd")
        if cd is None: continue
        
        # Depth oscillations: sign changes in direction of change
        up_depth = t.get("up_depth") or t.get("up_bid_depth")
        dn_depth = t.get("dn_depth") or t.get("dn_bid_depth")
        if up_depth is not None and state["prev_up_depth"] is not None:
            d = up_depth - state["prev_up_depth"]
            dir_now = 1 if d > 0 else -1 if d < 0 else 0
            if state["prev_up_depth_dir"] is not None and dir_now != 0 and dir_now != state["prev_up_depth_dir"]:
                state["depth_osc_up"] += 1
            if dir_now != 0: state["prev_up_depth_dir"] = dir_now
        state["prev_up_depth"] = up_depth if up_depth is not None else state["prev_up_depth"]
        
        if dn_depth is not None and state["prev_dn_depth"] is not None:
            d = dn_depth - state["prev_dn_depth"]
            dir_now = 1 if d > 0 else -1 if d < 0 else 0
            if state["prev_dn_depth_dir"] is not None and dir_now != 0 and dir_now != state["prev_dn_depth_dir"]:
                state["depth_osc_dn"] += 1
            if dir_now != 0: state["prev_dn_depth_dir"] = dir_now
        state["prev_dn_depth"] = dn_depth if dn_depth is not None else state["prev_dn_depth"]
        
        # Lead changes: who is > 0.50?
        up_ask = t.get("up_ask") or t.get("up_bid")
        dn_ask = t.get("dn_ask") or t.get("dn_bid")
        if up_ask is not None and dn_ask is not None:
            lead = "UP" if up_ask > dn_ask else "DN" if dn_ask > up_ask else state["prev_lead"]
            if state["prev_lead"] is not None and lead != state["prev_lead"]:
                state["lead_changes"] += 1
            state["prev_lead"] = lead
            # Time near 50/50
            if abs(up_ask - 0.50) < 0.10 and abs(dn_ask - 0.50) < 0.10:
                # Rough: add 1 second per tick in this range
                if state["prev_cd"] is not None:
                    dt = state["prev_cd"] - cd
                    if 0 < dt < 5:
                        state["time_near_5050"] += dt
        
        # BN tracking
        bn = t.get("bn_delta") or t.get("bn")
        if bn is not None:
            if state["first_bn_sign"] is None and bn != 0:
                state["first_bn_sign"] = 1 if bn > 0 else -1
            if state["prev_bn"] is not None:
                if (state["prev_bn"] > 0) != (bn > 0) and abs(bn) > 0.001 and abs(state["prev_bn"]) > 0.001:
                    state["bn_flips"] += 1
            if state["first_bn_sign"] is not None:
                cur = 1 if bn > 0 else -1 if bn < 0 else 0
                if cur != 0 and cur != state["first_bn_sign"]:
                    state["bn_flipped_sign"] = True
            state["prev_bn"] = bn
            state["bn_max_abs"] = max(state["bn_max_abs"], abs(bn))
            if abs(bn) > 0:
                state["bn_min_abs"] = min(state["bn_min_abs"], abs(bn))
        
        state["prev_cd"] = cd
        
        # Take a snapshot
        snapshots.append((cd, {
            "depth_osc_up":     state["depth_osc_up"],
            "depth_osc_dn":     state["depth_osc_dn"],
            "depth_osc_total":  state["depth_osc_up"] + state["depth_osc_dn"],
            "lead_changes":     state["lead_changes"],
            "time_near_5050":   state["time_near_5050"],
            "bn_flips":         state["bn_flips"],
            "bn_flipped_sign":  state["bn_flipped_sign"],
            "bn_max_abs":       state["bn_max_abs"],
            "bn_abs_range":     state["bn_max_abs"] - state["bn_min_abs"],
        }))
    
    return snapshots

def snapshot_at_cd(snapshots, target_cd):
    """Find the snapshot closest to target cd (but not earlier than target)."""
    if not snapshots: return None
    # Snapshots are sorted by cd descending (from ~300 down to ~0)
    # Find the LAST snapshot with cd >= target_cd
    best = None
    for cd, s in snapshots:
        if cd >= target_cd:
            best = (cd, s)
        else:
            break
    return best

# ═══════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def pearson_r(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = sum((xs[i]-mx)**2 for i in range(n))
    dy = sum((ys[i]-my)**2 for i in range(n))
    if dx == 0 or dy == 0: return 0.0
    return num / math.sqrt(dx*dy)

def classify_regime(features):
    """Classify CHOPPY / TRENDING / NEUTRAL from feature values."""
    if features["depth_osc_total"] >= 350 and features["bn_flipped_sign"]:
        return "CHOPPY"
    if features["depth_osc_total"] <= 200 and not features["bn_flipped_sign"]:
        return "TRENDING"
    return "NEUTRAL"

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to market_history.jsonl")
    ap.add_argument("--checkpoint-cds", default="200,150,100,50",
                    help="Comma-separated cd values to snapshot at")
    ap.add_argument("--limit", type=int, default=0, help="Process only N markets (0=all)")
    args = ap.parse_args()
    
    checkpoints = [int(c) for c in args.checkpoint_cds.split(",")]
    
    print("="*90)
    print("  REGIME TICK ANALYZER — Cumulative Feature Persistence Test")
    print("="*90)
    print(f"  Checkpoint cds: {checkpoints}")
    print(f"  Input: {args.path}")
    
    # Collect per-market: end-of-market features + each checkpoint features
    per_market = []
    n_processed = 0
    n_skipped = 0
    
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                market = json.loads(line)
            except Exception as e:
                n_skipped += 1
                continue
            
            snapshots = compute_cumulative_features(market)
            if not snapshots:
                n_skipped += 1
                continue
            
            end_feat = snapshots[-1][1]  # last snapshot is at lowest cd (closest to settle)
            
            row = dict(slug=market.get("slug"), end=end_feat, checkpoints={})
            for cp_cd in checkpoints:
                snap = snapshot_at_cd(snapshots, cp_cd)
                if snap:
                    row["checkpoints"][cp_cd] = snap[1]
            per_market.append(row)
            n_processed += 1
            
            if args.limit and n_processed >= args.limit: break
    
    print(f"  Processed: {n_processed} markets   Skipped: {n_skipped}")
    
    if n_processed < 10:
        print("  ❌ Not enough data. Need at least 10 markets with tick data.")
        sys.exit(1)
    
    # ═══ Persistence analysis ═══
    print("\n" + "="*90)
    print("  PERSISTENCE: does feature at checkpoint cd predict end-of-market value?")
    print("="*90)
    print("\n  Pearson r: 1.0=perfect persistence, 0.0=no predictive value")
    print(f"  {'checkpoint':>10} | " + " | ".join(f"{feat:>15}" for feat in 
          ["depth_osc_total", "lead_changes", "time_5050", "bn_flips", "bn_max_abs"]))
    print("  " + "─" * 90)
    
    for cp_cd in checkpoints:
        row = f"  cd={cp_cd:>3}     | "
        for feat in ["depth_osc_total", "lead_changes", "time_near_5050", "bn_flips", "bn_max_abs"]:
            xs = []; ys = []
            for m in per_market:
                cp = m["checkpoints"].get(cp_cd)
                if cp is None: continue
                xv = cp.get(feat); yv = m["end"].get(feat)
                if xv is None or yv is None: continue
                xs.append(xv); ys.append(yv)
            r = pearson_r(xs, ys) if len(xs) > 5 else 0.0
            row += f"{r:>+14.3f}  | "
        print(row)
    
    # ═══ Classification accuracy ═══
    print("\n" + "="*90)
    print("  REGIME CLASSIFICATION: does regime at checkpoint match end-of-market regime?")
    print("="*90)
    print(f"\n  Classification: CHOPPY (osc≥350 AND bn_flipped), TRENDING (osc≤200 AND NOT bn_flipped), else NEUTRAL")
    print(f"\n  {'checkpoint':>10} | {'total':>5} {'match':>5} {'accuracy':>8} {'CHOPPY→CHOPPY':>15} {'TREND→TREND':>13}")
    
    for cp_cd in checkpoints:
        total = match = 0
        c_right = t_right = 0
        c_total = t_total = 0
        for m in per_market:
            cp = m["checkpoints"].get(cp_cd)
            if cp is None: continue
            end_regime = classify_regime(m["end"])
            cp_regime = classify_regime(cp)
            total += 1
            if end_regime == cp_regime: match += 1
            if end_regime == "CHOPPY":
                c_total += 1
                if cp_regime == "CHOPPY": c_right += 1
            if end_regime == "TRENDING":
                t_total += 1
                if cp_regime == "TRENDING": t_right += 1
        acc = match/max(total,1)
        c_acc = c_right/max(c_total,1)
        t_acc = t_right/max(t_total,1)
        print(f"  cd={cp_cd:>3}     | {total:>5} {match:>5} {acc:>7.1%} "
              f"{c_right}/{c_total} ({c_acc:.0%}){' ':>5} {t_right}/{t_total} ({t_acc:.0%})")
    
    # ═══ Save raw data ═══
    out_path = "regime_persistence_data.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_markets": n_processed,
            "checkpoints": checkpoints,
            "data": [
                {"slug": m["slug"],
                 "end": m["end"],
                 "checkpoints": {str(k): v for k, v in m["checkpoints"].items()}}
                for m in per_market
            ]
        }, f)
    print(f"\n  Raw data saved: {out_path}")
    
    # ═══ Bottom-line: is this useful? ═══
    print("\n" + "="*90)
    print("  INTERPRETATION")
    print("="*90)
    # Grab the cd=150 row as a representative
    sample_cd = 150 if 150 in checkpoints else checkpoints[len(checkpoints)//2]
    print(f"\n  At cd={sample_cd} (halfway through market, typical fire time):")
    total = match = 0
    for m in per_market:
        cp = m["checkpoints"].get(sample_cd)
        if cp is None: continue
        end_regime = classify_regime(m["end"])
        cp_regime = classify_regime(cp)
        total += 1
        if end_regime == cp_regime: match += 1
    if total:
        acc = match/total
        print(f"    Regime classification accuracy: {acc:.1%} (baseline: random 33%)")
        if acc > 0.55:
            print(f"    ✅ STRONG SIGNAL — regime is predictable at fire time.")
            print(f"    Recommend: build regime_router.py that uses mid-market state to route strategies.")
        elif acc > 0.45:
            print(f"    🟡 MODEST SIGNAL — some predictability, needs finer thresholds.")
        else:
            print(f"    ❌ NO SIGNAL — regime is unpredictable mid-market.")
            print(f"    Regime routing at fire time won't work. Use end-of-market regime only for post-hoc learning.")
    
    print("="*90)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
