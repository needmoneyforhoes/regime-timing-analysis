#!/usr/bin/env python3
"""
post_fix_analysis.py — analyzes performance since the WS lag fix.

The WS lag/disk thrash issue was fixed by rotating market_history.jsonl
and deploying the auto-rotate hook. Before that, the bot was operating
on stale prices ("ws lag") which contaminated decisions and PnL.

This script answers three real questions:

1. Has performance changed since the fix? (compare pre-fix vs post-fix windows)
2. Is auto-demote actually removing losers? (check live_allowlist evolution)
3. What's the "trending market" loss pattern? (find markets where most fires lost)

Run from ./.
"""
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime


# Approximate timestamp when the WS fix was deployed
# Adjust if you remember a more specific time
WS_FIX_CUTOFF_SEC_AGO = 4 * 3600  # last 4 hours = post-fix


def main():
    cutoff_ts = time.time() - WS_FIX_CUTOFF_SEC_AGO
    
    print()
    print("═" * 100)
    print("  POST-FIX ANALYSIS — has the bot improved since the WS lag fix?")
    print("═" * 100)
    print(f"\n  Cutoff: last {WS_FIX_CUTOFF_SEC_AGO/3600:.0f} hours = POST-FIX")
    print(f"  Anything before that = PRE-FIX (contaminated data)")
    print()
    
    # ─── 1. Load market_history.jsonl, split by cutoff ───
    print("─" * 100)
    print("[1] PER-MARKET PnL — pre-fix vs post-fix")
    print("─" * 100)
    
    if not os.path.exists("market_history.jsonl"):
        print("  ✗ market_history.jsonl not found")
        return 1
    
    pre_fix_markets = []
    post_fix_markets = []
    with open("market_history.jsonl") as f:
        for line in f:
            try:
                m = json.loads(line)
                ts = m.get('settle_ts', m.get('ts', 0))
                slug = m.get('slug', '?')
                
                # Try multiple PnL field names (depends on bot version)
                hypo_pnl = (m.get('total_hypo_pnl') or 
                           m.get('hypo_pnl_total') or
                           m.get('this_market_hypo_gc') or
                           m.get('hypo_gc') or 0)
                
                # If still 0, try summing fires
                if hypo_pnl == 0:
                    fires = m.get('fires', [])
                    if fires:
                        hypo_pnl = sum(f.get('hypo_pnl', 0) for f in fires)
                
                n_fires = len(m.get('fires', []))
                winner = m.get('winner', '?')
                
                rec = {'ts': ts, 'slug': slug, 'pnl': hypo_pnl, 
                       'n_fires': n_fires, 'winner': winner}
                
                if ts >= cutoff_ts:
                    post_fix_markets.append(rec)
                else:
                    pre_fix_markets.append(rec)
            except Exception:
                continue
    
    print(f"\n  Pre-fix markets:  {len(pre_fix_markets)}")
    print(f"  Post-fix markets: {len(post_fix_markets)}")
    
    if len(post_fix_markets) == 0:
        print(f"\n  ✗ No post-fix markets found. Adjust WS_FIX_CUTOFF_SEC_AGO at top of script.")
        # Try to show what timestamps are in the file
        if pre_fix_markets:
            recent_ts = max(m['ts'] for m in pre_fix_markets)
            print(f"  Most recent timestamp in file: {datetime.fromtimestamp(recent_ts)}")
            print(f"  Hours ago: {(time.time() - recent_ts) / 3600:.1f}h")
        return 1
    
    # Average per-market PnL
    pre_avg = sum(m['pnl'] for m in pre_fix_markets) / max(len(pre_fix_markets), 1)
    post_avg = sum(m['pnl'] for m in post_fix_markets) / max(len(post_fix_markets), 1)
    pre_total = sum(m['pnl'] for m in pre_fix_markets)
    post_total = sum(m['pnl'] for m in post_fix_markets)
    
    print(f"\n  PRE-FIX:")
    print(f"    Total hypo PnL:   ${pre_total:+.2f}")
    print(f"    Avg per market:   ${pre_avg:+.2f}/market")
    print(f"    Markets:          {len(pre_fix_markets)}")
    
    print(f"\n  POST-FIX:")
    print(f"    Total hypo PnL:   ${post_total:+.2f}")
    print(f"    Avg per market:   ${post_avg:+.2f}/market")
    print(f"    Markets:          {len(post_fix_markets)}")
    
    delta = post_avg - pre_avg
    pct = (delta / abs(pre_avg) * 100) if pre_avg != 0 else 0
    
    print(f"\n  CHANGE: ${delta:+.2f}/market ({pct:+.1f}%)")
    if delta > 1.0:
        print(f"  ✅ MEANINGFUL IMPROVEMENT")
    elif delta > 0:
        print(f"  🟡 Slight improvement (within noise)")
    elif delta > -1.0:
        print(f"  ➡️  Roughly unchanged (within noise)")
    else:
        print(f"  ❌ DEGRADATION")
    
    # Statistical test: is the difference significant?
    # Use a simple t-test approximation
    if len(pre_fix_markets) >= 5 and len(post_fix_markets) >= 5:
        import statistics
        pre_pnls = [m['pnl'] for m in pre_fix_markets]
        post_pnls = [m['pnl'] for m in post_fix_markets]
        pre_std = statistics.stdev(pre_pnls) if len(pre_pnls) > 1 else 1
        post_std = statistics.stdev(post_pnls) if len(post_pnls) > 1 else 1
        # Welch's t (rough)
        se = ((pre_std**2 / len(pre_pnls)) + (post_std**2 / len(post_pnls))) ** 0.5
        if se > 0:
            t = delta / se
            print(f"\n  Welch's t ≈ {t:.2f} (|t|>2 = stat significant)")
            if abs(t) >= 2:
                print(f"  ✅ Difference is statistically meaningful (not just noise)")
            else:
                print(f"  ⚠️  Difference is within statistical noise")
    
    # ─── 2. Last 12 post-fix markets in detail ───
    print(f"\n─" * 100)
    print("[2] POST-FIX MARKETS — detail")
    print("─" * 100)
    print(f"\n  {'#':<4s} {'WHEN':<22s} {'SLUG':<35s} {'WINNER':<8s} {'FIRES':<7s} {'HYPO PnL':<12s}")
    
    for i, m in enumerate(post_fix_markets[-15:], 1):  # last 15
        when = datetime.fromtimestamp(m['ts']).strftime('%m-%d %H:%M:%S')
        pnl_str = f"${m['pnl']:+.2f}"
        emoji = "✅" if m['pnl'] > 0 else ("➡️" if m['pnl'] == 0 else "❌")
        print(f"  {i:<4d} {when:<22s} {m['slug']:<35s} {m['winner']:<8s} {m['n_fires']:<7d} {pnl_str:<12s} {emoji}")
    
    # Win rate
    wins = sum(1 for m in post_fix_markets if m['pnl'] > 0)
    losses = sum(1 for m in post_fix_markets if m['pnl'] < 0)
    flat = sum(1 for m in post_fix_markets if m['pnl'] == 0)
    print(f"\n  Win rate (per market): {wins}/{len(post_fix_markets)} = {wins/max(len(post_fix_markets),1)*100:.1f}%")
    print(f"  Markets won:    {wins}")
    print(f"  Markets lost:   {losses}")
    print(f"  Markets flat:   {flat}")
    
    # ─── 3. Auto-demote activity ───
    print(f"\n─" * 100)
    print("[3] AUTO-DEMOTE — is the system removing losers?")
    print("─" * 100)
    
    log_path = "auto_improvements.log"
    induction_path = "induction_train.out"
    
    promoted_total = 0
    demoted_total = 0
    promoted_recent = 0
    demoted_recent = 0
    
    # Look in induction_train.out for both events
    if os.path.exists(induction_path):
        with open(induction_path) as f:
            for line in f:
                if 'PROMOTED' in line and 'FLIP' in line:
                    promoted_total += 1
                if 'DEMOTED' in line:
                    demoted_total += 1
        # Also check just the recent portion (last 500 lines = recent activity)
        with open(induction_path) as f:
            lines = f.readlines()
        recent_lines = lines[-500:] if len(lines) > 500 else lines
        for line in recent_lines:
            if 'PROMOTED' in line and 'FLIP' in line:
                promoted_recent += 1
            if 'DEMOTED' in line:
                demoted_recent += 1
    
    print(f"\n  Lifetime activity in induction_train.out:")
    print(f"    Promotions: {promoted_total}")
    print(f"    Demotions:  {demoted_total}")
    print(f"\n  Recent activity (last 500 log lines):")
    print(f"    Promotions: {promoted_recent}")
    print(f"    Demotions:  {demoted_recent}")
    
    if demoted_total == 0:
        print(f"\n  ⚠️  AUTO-DEMOTE HAS NEVER FIRED")
        print(f"     This means: no strategies have been removed from LIVE allowlist.")
        print(f"     Possible reasons:")
        print(f"       - Geo-blocked: no LIVE-mode fires to evaluate (most likely)")
        print(f"       - LIVE strategies haven't accumulated enough fires to trigger")
        print(f"       - All LIVE strategies are still passing the demote criteria")
    elif demoted_recent == 0:
        print(f"\n  🟡 Demote system worked historically but not recently")
    else:
        print(f"\n  ✅ Auto-demote actively removing losers")
    
    # ─── 4. Trending market analysis ───
    print(f"\n─" * 100)
    print("[4] TRENDING MARKET LOSS PATTERN")
    print("─" * 100)
    print(f"\n  Looking for markets where most fires lost (one-side-dominant markets):")
    
    bad_markets = []  # Markets where >70% of fires lost
    for m in post_fix_markets[-30:]:  # last 30 post-fix
        if m['n_fires'] < 5:
            continue  # need enough fires to evaluate
        # Estimate from PnL: if very negative with many fires, fires lost
        if m['n_fires'] > 0 and m['pnl'] < -m['n_fires'] * 1.0:
            # Roughly: most fires lost
            bad_markets.append(m)
    
    if bad_markets:
        print(f"\n  {'WHEN':<20s} {'SLUG':<35s} {'WINNER':<8s} {'FIRES':<7s} {'PnL':<12s}")
        for m in bad_markets[-10:]:
            when = datetime.fromtimestamp(m['ts']).strftime('%m-%d %H:%M:%S')
            print(f"  {when:<20s} {m['slug']:<35s} {m['winner']:<8s} {m['n_fires']:<7d} ${m['pnl']:+.2f}")
        print(f"\n  Found {len(bad_markets)} trending-loss markets in last 30")
        print(f"  These are markets where many fires happened but most lost.")
        print(f"  This matches your 'cheap entry strategies in directional markets' theory.")
    else:
        print(f"\n  No clearly trending-loss markets in last 30 post-fix markets.")
    
    # ─── 5. Live allowlist current state ───
    print(f"\n─" * 100)
    print("[5] LIVE ALLOWLIST EVOLUTION")
    print("─" * 100)
    
    if os.path.exists("live_allowlist.json"):
        with open("live_allowlist.json") as f:
            allowlist = json.load(f)
        n_live = len(allowlist) if isinstance(allowlist, list) else len(allowlist)
        mtime_age = (time.time() - os.path.getmtime("live_allowlist.json")) / 60
        print(f"\n  Current allowlist: {n_live} strategies")
        print(f"  Last modified: {mtime_age:.0f}m ago")
        if mtime_age > 60 and len(post_fix_markets) > 5:
            print(f"  ⚠️  Allowlist hasn't updated since fix — auto-promote may not have fired post-fix")
    
    # ─── Final verdict ───
    print(f"\n═" * 100)
    print("  POST-FIX VERDICT")
    print("═" * 100)
    
    issues = []
    positives = []
    
    if delta > 1.0:
        positives.append(f"Per-market PnL improved by ${delta:+.2f}")
    elif delta < -1.0:
        issues.append(f"Per-market PnL got WORSE by ${delta:+.2f}")
    
    if post_avg > 0:
        positives.append(f"Post-fix avg is POSITIVE: ${post_avg:+.2f}/market")
    elif post_avg < -3.0:
        issues.append(f"Post-fix avg still significantly negative: ${post_avg:+.2f}/market")
    
    win_pct = wins/max(len(post_fix_markets),1)*100
    if win_pct > 55:
        positives.append(f"Per-market win rate > 55%: {win_pct:.1f}%")
    elif win_pct < 40:
        issues.append(f"Per-market win rate < 40%: {win_pct:.1f}%")
    
    if demoted_total == 0:
        issues.append("Auto-demote has never fired (no losers being cut)")
    
    if len(bad_markets) > 5:
        issues.append(f"Trending-market losses still happening ({len(bad_markets)} of last 30)")
    
    if positives:
        print(f"\n  ✅ POSITIVES:")
        for p in positives:
            print(f"     • {p}")
    
    if issues:
        print(f"\n  ⚠️  ISSUES:")
        for iss in issues:
            print(f"     • {iss}")
    
    print()
    if post_avg > 0 and not issues:
        print(f"  ✅ Bot is profitable post-fix. Consider canary deploy.")
    elif post_avg > 0:
        print(f"  🟡 Bot is profitable but has open issues. Investigate before deploy.")
    elif post_avg > -2:
        print(f"  ⚠️  Roughly break-even. Marginal — would lose to fees/slippage in LIVE.")
    else:
        print(f"  ❌ Still losing meaningfully. Don't deploy yet.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
