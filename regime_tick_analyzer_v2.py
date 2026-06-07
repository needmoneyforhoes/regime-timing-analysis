#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_tick_analyzer_v2.py — Run on VPS against market_history.jsonl

FIXED to use actual schema:
  tick_columns = list of column NAMES (27 cols)
  ticks = list of rows (each row = list of values matching tick_columns order)

Key columns used:
  cd               — countdown to settle (300 down to 0)
  up_ask, dn_ask   — asking prices (for lead detection)
  up_bid, dn_bid   — bid prices
  bn_delta_pct     — order-book imbalance (THE signal, in percent)
  up_depth, dn_depth — order book depth

WHAT THIS ANSWERS:

  Q1 (persistence): at cd=150 (typical fire time), do cumulative market
      features predict end-of-market regime?

  Q2 (late-flip detectability): can we detect BN sign flips in the final
      120 seconds of a market? This is the Option B protection.

  Q3 (disaster signature): what tick-level features precede disaster
      markets (barbell loss ≥ $5)?

USAGE:
  python3 regime_tick_analyzer_v2.py market_history.jsonl
  python3 regime_tick_analyzer_v2.py market_history.jsonl --limit 50

OUTPUT FILES:
  regime_tick_report.txt        — human-readable
  regime_persistence_data.json  — raw numbers for further analysis
"""

import json
import sys
import argparse
import math
import statistics
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
#  SCHEMA — these are the REAL column names from VPS
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = ["cd", "up_ask", "dn_ask", "up_bid", "dn_bid",
                 "up_depth", "dn_depth", "bn_delta_pct",
                 "crowd_side", "crowd_conviction", "cl_delta"]

# ═══════════════════════════════════════════════════════════════════════════
#  TICK STREAM PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_ticks(market):
    """Return sorted-descending-by-cd list of tick dicts."""
    cols = market.get("tick_columns") or []
    rows = market.get("ticks") or []
    if not cols or not rows:
        return []

    col_idx = {c: i for i, c in enumerate(cols)}
    missing = [c for c in REQUIRED_COLS if c not in col_idx]
    if missing:
        # Don't fail — just skip missing fields; some markets may have reduced schema
        pass

    ticks = []
    for row in rows:
        if not isinstance(row, list): continue
        d = {}
        for c, i in col_idx.items():
            if i < len(row):
                d[c] = row[i]
        if d.get("cd") is None: continue
        ticks.append(d)

    # cd counts DOWN, so sort by cd descending (market start first)
    ticks.sort(key=lambda t: -t["cd"])
    return ticks

# ═══════════════════════════════════════════════════════════════════════════
#  CUMULATIVE FEATURE ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_snapshots(ticks, checkpoint_cds):
    """Walk the tick stream and record cumulative features at each checkpoint cd.

    Features tracked:
      depth_osc_up:     count of up_depth direction reversals
      depth_osc_dn:     count of dn_depth direction reversals
      lead_changes:     count of times which side was leading (>0.50 ask) flipped
      time_near_5050:   cumulative seconds both asks within 0.40-0.60
      bn_flips:         count of bn_delta_pct sign reversals (|bn|>0.01 threshold)
      bn_crossed_zero:  1 if bn crossed from positive to negative (or vice versa)
      bn_abs_max:       max |bn_delta_pct|
      bn_abs_range:     max - min of |bn_delta_pct|
    """
    state = dict(
        depth_osc_up=0, depth_osc_dn=0,
        lead_changes=0, time_near_5050=0.0,
        bn_flips=0, bn_crossed_zero=False,
        bn_abs_max=0.0, bn_abs_min=1.0,
        prev_up_depth=None, prev_dn_depth=None,
        prev_up_depth_dir=None, prev_dn_depth_dir=None,
        prev_lead=None,  # +1 if UP leading (dn_ask > up_ask), -1 if DN
        prev_bn=None, first_bn_sign=None,
        prev_cd=None,
    )

    snapshots = {cp: None for cp in checkpoint_cds}
    end_snap = None

    for t in ticks:
        cd = t["cd"]

        # Depth oscillations
        ud = t.get("up_depth")
        if ud is not None and state["prev_up_depth"] is not None:
            diff = ud - state["prev_up_depth"]
            dir_ = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if dir_ != 0:
                if state["prev_up_depth_dir"] is not None and dir_ != state["prev_up_depth_dir"]:
                    state["depth_osc_up"] += 1
                state["prev_up_depth_dir"] = dir_
        if ud is not None: state["prev_up_depth"] = ud

        dd = t.get("dn_depth")
        if dd is not None and state["prev_dn_depth"] is not None:
            diff = dd - state["prev_dn_depth"]
            dir_ = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if dir_ != 0:
                if state["prev_dn_depth_dir"] is not None and dir_ != state["prev_dn_depth_dir"]:
                    state["depth_osc_dn"] += 1
                state["prev_dn_depth_dir"] = dir_
        if dd is not None: state["prev_dn_depth"] = dd

        # Lead changes (based on ASK prices — whoever has higher ask is "leading" the win)
        ua = t.get("up_ask"); da = t.get("dn_ask")
        if ua is not None and da is not None:
            if ua > da: lead = 1
            elif da > ua: lead = -1
            else: lead = state["prev_lead"]
            if state["prev_lead"] is not None and lead != state["prev_lead"] and lead is not None:
                state["lead_changes"] += 1
            if lead is not None: state["prev_lead"] = lead
            # Time near 50/50: both asks in [0.40, 0.60]
            if 0.40 <= ua <= 0.60 and 0.40 <= da <= 0.60:
                if state["prev_cd"] is not None:
                    dt = state["prev_cd"] - cd
                    if 0 < dt < 5:  # sanity — typical tick gap 0.1-2s
                        state["time_near_5050"] += dt

        # BN tracking
        bn = t.get("bn_delta_pct")
        if bn is not None:
            abs_bn = abs(bn)
            if abs_bn > state["bn_abs_max"]: state["bn_abs_max"] = abs_bn
            if abs_bn > 0 and abs_bn < state["bn_abs_min"]: state["bn_abs_min"] = abs_bn
            if state["first_bn_sign"] is None and abs_bn > 0.005:
                state["first_bn_sign"] = 1 if bn > 0 else -1
            if state["prev_bn"] is not None:
                # Detect sign flip — require both magnitudes to exceed threshold to avoid noise
                if (state["prev_bn"] > 0) != (bn > 0) and abs(state["prev_bn"]) > 0.01 and abs_bn > 0.01:
                    state["bn_flips"] += 1
            if state["first_bn_sign"] is not None and abs_bn > 0.01:
                cur_sign = 1 if bn > 0 else -1
                if cur_sign != state["first_bn_sign"]:
                    state["bn_crossed_zero"] = True
            state["prev_bn"] = bn

        state["prev_cd"] = cd

        # Capture snapshot at checkpoints — take the MOST RECENT tick with cd >= checkpoint
        for cp in checkpoint_cds:
            if cd >= cp:
                snapshots[cp] = _snap_state(state, bn_now=bn)

        end_snap = _snap_state(state, bn_now=bn)

    return snapshots, end_snap

def _snap_state(state, bn_now=None):
    return dict(
        depth_osc_up=state["depth_osc_up"],
        depth_osc_dn=state["depth_osc_dn"],
        depth_osc_total=state["depth_osc_up"] + state["depth_osc_dn"],
        lead_changes=state["lead_changes"],
        time_near_5050=state["time_near_5050"],
        bn_flips=state["bn_flips"],
        bn_crossed_zero=state["bn_crossed_zero"],
        bn_abs_max=state["bn_abs_max"],
        bn_now=bn_now,
        bn_now_abs=abs(bn_now) if bn_now is not None else 0.0,
        bn_now_sign=(1 if (bn_now or 0) > 0 else (-1 if (bn_now or 0) < 0 else 0)),
    )

# ═══════════════════════════════════════════════════════════════════════════
#  LATE BN FLIP DETECTION (Option B validation)
# ═══════════════════════════════════════════════════════════════════════════

def detect_late_bn_flip(ticks, early_cd=120, late_cd=10):
    """Compare BN sign at early_cd vs late_cd. Returns dict with diagnostics."""
    if not ticks: return None

    # Find tick closest to cd=120 (or just after)
    early_bn = None; early_cd_actual = None
    late_bn = None; late_cd_actual = None

    for t in ticks:
        cd = t["cd"]; bn = t.get("bn_delta_pct")
        if bn is None: continue
        if cd >= early_cd and (early_cd_actual is None or abs(cd-early_cd) < abs(early_cd_actual-early_cd)):
            early_bn = bn; early_cd_actual = cd
        if cd <= late_cd and late_cd_actual is None:
            late_bn = bn; late_cd_actual = cd

    if early_bn is None or late_bn is None:
        return None

    # Require meaningful magnitudes (avoid zero-crossing noise)
    meaningful = abs(early_bn) > 0.01 and abs(late_bn) > 0.01
    flipped = meaningful and ((early_bn > 0) != (late_bn > 0))

    # Also track: did BN flip at any point BETWEEN early and late?
    any_flip_between = False
    between_ticks = [t for t in ticks if late_cd < t["cd"] < early_cd]
    last_sign = None
    for t in between_ticks:
        bn = t.get("bn_delta_pct")
        if bn is None or abs(bn) < 0.01: continue
        s = 1 if bn > 0 else -1
        if last_sign is not None and s != last_sign:
            any_flip_between = True
        last_sign = s

    return dict(
        early_cd=early_cd_actual, early_bn=early_bn,
        late_cd=late_cd_actual, late_bn=late_bn,
        flipped=flipped,
        any_flip_between=any_flip_between,
        meaningful=meaningful,
    )

# ═══════════════════════════════════════════════════════════════════════════
#  STATS HELPERS
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

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--checkpoints", default="200,150,120,100,60,30",
                    help="comma-separated cd values")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="regime_tick_report.txt")
    args = ap.parse_args()

    checkpoints = [int(c) for c in args.checkpoints.split(",")]

    out_lines = []
    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("=" * 100)
    emit("  REGIME TICK ANALYZER v2  —  Fixed to use real column schema")
    emit("=" * 100)
    emit(f"  Input:       {args.path}")
    emit(f"  Checkpoints: cd={checkpoints}")
    emit(f"  Limit:       {args.limit or 'no limit'}")

    markets_data = []
    n_read = 0; n_ok = 0; n_no_ticks = 0

    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            n_read += 1
            ticks = parse_ticks(m)
            if not ticks:
                n_no_ticks += 1
                if args.limit and n_ok >= args.limit: break
                continue

            snapshots, end_snap = compute_snapshots(ticks, checkpoints)
            flip_info = detect_late_bn_flip(ticks, early_cd=120, late_cd=10)

            markets_data.append({
                "slug": m.get("slug"),
                "winner": m.get("winner"),
                "n_ticks": len(ticks),
                "snapshots": snapshots,
                "end_snap": end_snap,
                "late_flip": flip_info,
                "bn_delta_final": m.get("bn_delta_final"),
            })
            n_ok += 1
            if args.limit and n_ok >= args.limit: break

    emit(f"\n  Read: {n_read} markets   OK: {n_ok}   No-ticks: {n_no_ticks}")

    if n_ok < 20:
        emit("  ❌ Not enough data")
        with open(args.out, "w") as f: f.write("\n".join(out_lines))
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # Q1: PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════
    emit("\n" + "=" * 100)
    emit("  Q1: PERSISTENCE  —  does feature at checkpoint predict end-of-market value?")
    emit("=" * 100)
    emit("\n  Pearson r: 1.0 = perfect predictor, 0.0 = no signal")
    emit(f"  {'feature':>18} | " + " | ".join(f"{'cd='+str(c):>9}" for c in checkpoints))
    emit("  " + "─" * 100)

    feat_order = ["depth_osc_total", "lead_changes", "time_near_5050", "bn_flips", "bn_abs_max"]
    for feat in feat_order:
        row = f"  {feat:>18} | "
        for cp in checkpoints:
            xs, ys = [], []
            for md in markets_data:
                snap = md["snapshots"].get(cp)
                if snap is None: continue
                end = md["end_snap"]
                xv = snap.get(feat); yv = end.get(feat)
                if xv is None or yv is None: continue
                xs.append(xv); ys.append(yv)
            r = pearson_r(xs, ys) if len(xs) > 10 else 0.0
            tag = "★" if abs(r) > 0.6 else ("·" if abs(r) > 0.4 else " ")
            row += f" {r:>+6.3f}{tag:>2} |"
        emit(row)

    emit("\n  Legend: ★ = strong predictor (|r|>0.6)   · = modest predictor (|r|>0.4)")

    # ═══════════════════════════════════════════════════════════════════
    # Q2: LATE BN FLIP DETECTABILITY (Option B validation)
    # ═══════════════════════════════════════════════════════════════════
    emit("\n" + "=" * 100)
    emit("  Q2: LATE BN FLIP  —  how often does BN sign differ between cd=120 and cd=10?")
    emit("=" * 100)

    with_flip = [md for md in markets_data if md["late_flip"] and md["late_flip"]["meaningful"]]
    flipped = [md for md in with_flip if md["late_flip"]["flipped"]]
    any_intra = [md for md in with_flip if md["late_flip"]["any_flip_between"]]

    emit(f"\n  Markets analyzed:               {n_ok}")
    emit(f"  Markets with meaningful BN:     {len(with_flip)}  ({len(with_flip)/n_ok*100:.0f}%)")
    emit(f"  BN sign flipped T-120 → T-10:   {len(flipped)}  ({len(flipped)/max(len(with_flip),1)*100:.0f}% of meaningful)")
    emit(f"  ANY intermediate flip:          {len(any_intra)}  ({len(any_intra)/max(len(with_flip),1)*100:.0f}% of meaningful)")

    # Show examples
    emit("\n  Sample of flip markets (first 10):")
    for md in flipped[:10]:
        lf = md["late_flip"]
        emit(f"    {md['slug'][-20:]:>20}  winner={md['winner']}  "
             f"T-{lf['early_cd']:.0f}:BN={lf['early_bn']:+.4f}  →  "
             f"T-{lf['late_cd']:.0f}:BN={lf['late_bn']:+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Q3: DOES FLIP PREDICT WINNER REVERSAL?
    # ═══════════════════════════════════════════════════════════════════
    emit("\n" + "=" * 100)
    emit("  Q3: HEDGING USEFULNESS  —  when BN flips late, does the winner match the LATE sign?")
    emit("=" * 100)

    # If BN flipped from + at T-120 to - at T-10, and winner = DN, the LATE signal was correct.
    # If a strategy fired UP at T-120 based on the early signal, exiting at the late sign would save loss.

    flip_winner_matches_late = 0
    flip_winner_matches_early = 0
    for md in flipped:
        lf = md["late_flip"]
        w = md["winner"]
        early_sign = "UP" if lf["early_bn"] > 0 else "DN"
        late_sign = "UP" if lf["late_bn"] > 0 else "DN"
        if w == late_sign: flip_winner_matches_late += 1
        if w == early_sign: flip_winner_matches_early += 1

    if flipped:
        emit(f"\n  Of {len(flipped)} markets with BN flip T-120 → T-10:")
        emit(f"    Winner matched LATE BN sign:  {flip_winner_matches_late} ({flip_winner_matches_late/len(flipped)*100:.0f}%)")
        emit(f"    Winner matched EARLY BN sign: {flip_winner_matches_early} ({flip_winner_matches_early/len(flipped)*100:.0f}%)")
        emit(f"\n  Interpretation:")
        if flip_winner_matches_late > flip_winner_matches_early * 1.5:
            emit("    ✅ STRONG — late BN is a much better winner predictor than early BN")
            emit("    → Option B (exit on late flip) is HIGH-VALUE to implement")
        elif flip_winner_matches_late > flip_winner_matches_early:
            emit("    🟡 MODEST — late BN is moderately better than early BN")
            emit("    → Option B worth implementing with careful sizing")
        else:
            emit("    ❌ NO EDGE — late BN is not reliably better than early BN")
            emit("    → Option B may not protect against the flip scenario")

    # ═══════════════════════════════════════════════════════════════════
    # Q4: REGIME CLASSIFICATION ACCURACY AT FIRE TIME
    # ═══════════════════════════════════════════════════════════════════
    emit("\n" + "=" * 100)
    emit("  Q4: REGIME CLASSIFICATION  —  can we identify regime at fire time?")
    emit("=" * 100)
    emit("\n  CHOPPY: depth_osc_total ≥ 350 AND bn_crossed_zero")
    emit("  TRENDING: depth_osc_total ≤ 200 AND NOT bn_crossed_zero")
    emit("  NEUTRAL: else")

    def classify(s):
        if s["depth_osc_total"] >= 350 and s["bn_crossed_zero"]: return "CHOPPY"
        if s["depth_osc_total"] <= 200 and not s["bn_crossed_zero"]: return "TRENDING"
        return "NEUTRAL"

    emit(f"\n  {'checkpoint':>10} {'accuracy':>9} {'CHOPPY→CHOPPY':>16} {'TREND→TREND':>14} {'NEUTRAL→match':>15}")
    for cp in checkpoints:
        total = match = 0
        cm = tm = nm = 0
        ct = tt = nt = 0
        for md in markets_data:
            snap = md["snapshots"].get(cp)
            if snap is None: continue
            end_r = classify(md["end_snap"])
            cp_r = classify(snap)
            total += 1
            if end_r == cp_r: match += 1
            if end_r == "CHOPPY": ct += 1
            if end_r == "CHOPPY" and cp_r == "CHOPPY": cm += 1
            if end_r == "TRENDING": tt += 1
            if end_r == "TRENDING" and cp_r == "TRENDING": tm += 1
            if end_r == "NEUTRAL": nt += 1
            if end_r == "NEUTRAL" and cp_r == "NEUTRAL": nm += 1
        if total == 0: continue
        acc = match/total
        emit(f"  cd={cp:>3}     {acc:>7.1%}  {cm}/{ct}={cm/max(ct,1)*100:>3.0f}%         "
             f"{tm}/{tt}={tm/max(tt,1)*100:>3.0f}%         "
             f"{nm}/{nt}={nm/max(nt,1)*100:>3.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Save raw data
    # ═══════════════════════════════════════════════════════════════════
    out_json = "regime_persistence_data.json"
    with open(out_json, "w") as f:
        json.dump({"markets": [
            {"slug": md["slug"], "winner": md["winner"], "bn_delta_final": md["bn_delta_final"],
             "end_snap": md["end_snap"], "late_flip": md["late_flip"],
             "snapshots": {str(k): v for k, v in md["snapshots"].items()}}
            for md in markets_data
        ]}, f)
    emit(f"\n  Raw data saved: {out_json}")

    with open(args.out, "w") as f:
        f.write("\n".join(out_lines))
    emit(f"  Report saved:   {args.out}")
    emit("=" * 100)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
