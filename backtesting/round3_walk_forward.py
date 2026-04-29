"""
Leave-one-day-out walk-forward validation for Round 3 trader.

For each held-out day d:
  1. Run a random-search sweep on the OTHER two days (train).
  2. Take the best config from the train sweep.
  3. Score that config on day d (test).
  4. Compare:
       - default-params PnL on day d (baseline, no tuning)
       - all-3-days-tuned PnL on day d (overfit ceiling)
       - train-on-2-days-tuned PnL on day d (realistic OOS)
       - train PnL itself (what the sweep "thinks" we'll get)

The train→test PnL ratio is our shrinkage estimate.

Run: python3 -m backtesting.round3_walk_forward
"""
import sys
import time
import multiprocessing as mp
import random
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtesting.round3_sweep import (
    DEFAULT_PARAMS, _sample_random, PHASE2_RANGES, _run_one,
)

# Already-tuned config from the all-3-days sweep (writeups/round3_sweep_results.json).
ALL_DAYS_TUNED = dict(DEFAULT_PARAMS)
ALL_DAYS_TUNED.update({
    "GAMMA_HYDROGEL":      2.10484e-07,
    "GAMMA_VELVETFRUIT":   6.82375e-07,
    "MAX_HALF_SPREAD_HYDR": 9,
    "MAX_HALF_SPREAD_UND":  2,
    "MM_TOTAL_SIZE_HYDR":  178,
    "MM_TOTAL_SIZE_UND":   174,
    "VOUCHER_MIN_EDGE":    4.23952,
    "VOUCHER_MAX_TRADE":    25,
    "SIGMA_EWMA_ALPHA":    0.00257849,
    "DELTA_DEAD_ZONE":     75.4362,
})


def _run_with_days(args):
    """Worker: run trader.run_backtest with overrides on a specific subset of days."""
    params, days = args
    import importlib
    import submissions.round3.trader as t
    importlib.reload(t)
    from backtesting.round3_backtest import run_backtest

    t.GAMMA_HYDROGEL    = params["GAMMA_HYDROGEL"]
    t.GAMMA_VELVETFRUIT = params["GAMMA_VELVETFRUIT"]
    t.SIGMA_HYDROGEL    = params["SIGMA_HYDROGEL"]
    t.SIGMA_VELVETFRUIT = params["SIGMA_VELVETFRUIT"]
    t.K_HYDROGEL        = params["K_HYDROGEL"]
    t.K_VELVETFRUIT     = params["K_VELVETFRUIT"]
    t.MAX_HALF_SPREAD = {
        t.HYDROGEL:    int(params["MAX_HALF_SPREAD_HYDR"]),
        t.UNDERLYING:  int(params["MAX_HALF_SPREAD_UND"]),
    }
    t.MIN_HALF_SPREAD = {
        t.HYDROGEL:    int(params["MIN_HALF_SPREAD_HYDR"]),
        t.UNDERLYING:  int(params["MIN_HALF_SPREAD_UND"]),
    }
    t.MM_TOTAL_SIZE = {
        t.HYDROGEL:    int(params["MM_TOTAL_SIZE_HYDR"]),
        t.UNDERLYING:  int(params["MM_TOTAL_SIZE_UND"]),
    }
    t.MM_LEVELS              = int(params["MM_LEVELS"])
    t.VOUCHER_MIN_EDGE       = float(params["VOUCHER_MIN_EDGE"])
    t.VOUCHER_EDGE_FRAC_OF_SPREAD = float(params["VOUCHER_EDGE_FRAC"])
    t.VOUCHER_MAX_TRADE_SIZE = int(params["VOUCHER_MAX_TRADE"])
    t.SIGMA_EWMA_ALPHA       = float(params["SIGMA_EWMA_ALPHA"])
    t.DELTA_DEAD_ZONE        = float(params["DELTA_DEAD_ZONE"])
    return run_backtest(t.Trader(), passive_share=0.5, verbose=False, days=days)


def sweep_on_days(train_days, n_samples=300, seed=42, workers=8):
    """Random search restricted to `train_days`. Returns best (cfg, train_pnl)."""
    rng = random.Random(seed)
    configs = [_sample_random(rng) for _ in range(n_samples)]
    args = [(cfg, train_days) for cfg in configs]
    with mp.Pool(workers) as pool:
        pnls = pool.map(_run_with_days, args)
    best_idx = max(range(len(pnls)), key=lambda i: pnls[i])
    return configs[best_idx], pnls[best_idx]


def main():
    print("═" * 78)
    print("Walk-forward leave-one-day-out validation")
    print("═" * 78)
    print()

    all_days = [0, 1, 2]
    rows = []
    t0 = time.time()

    for held_out in all_days:
        train_days = [d for d in all_days if d != held_out]
        print(f"\n[Held-out day {held_out}, train on days {train_days}]")
        sweep_t0 = time.time()
        best_cfg, train_pnl = sweep_on_days(train_days, n_samples=300, seed=42 + held_out)
        sweep_dt = time.time() - sweep_t0
        # Score the train winner on held-out day
        oos_pnl = _run_with_days((best_cfg, [held_out]))
        # Score all-3-days-tuned on held-out (overfit ceiling)
        overfit_pnl = _run_with_days((ALL_DAYS_TUNED, [held_out]))
        # Score default-params on held-out (no tuning)
        default_pnl = _run_with_days((DEFAULT_PARAMS, [held_out]))
        rows.append({
            "held_out": held_out,
            "train_days": train_days,
            "train_pnl": train_pnl,
            "test_pnl": oos_pnl,
            "overfit_test_pnl": overfit_pnl,
            "default_test_pnl": default_pnl,
            "best_cfg": best_cfg,
        })
        print(f"  sweep done in {sweep_dt:.1f}s")
        print(f"  train_pnl       (2 days, tuned) = {train_pnl:>9.0f}")
        print(f"  test_pnl        (1 day, OOS)    = {oos_pnl:>9.0f}")
        print(f"  overfit_test    (1 day, all3-tuned) = {overfit_pnl:>9.0f}")
        print(f"  default_test    (1 day, defaults)   = {default_pnl:>9.0f}")
        ratio = oos_pnl / train_pnl if train_pnl > 0 else float("nan")
        print(f"  train→test ratio (per day) = {oos_pnl:.0f} / ({train_pnl:.0f} / 2) "
              f"= {ratio * 2 if train_pnl > 0 else float('nan'):.2f}")

    print(f"\nTotal walk-forward time: {time.time() - t0:.0f}s")

    # Aggregate summary
    print("\n" + "═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print(f"  {'held':>5}  {'train (2d)':>11}  {'OOS (1d)':>10}  "
          f"{'overfit-on-OOS':>15}  {'default OOS':>12}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['held_out']:>5}  "
              f"{r['train_pnl']:>11.0f}  {r['test_pnl']:>10.0f}  "
              f"{r['overfit_test_pnl']:>15.0f}  {r['default_test_pnl']:>12.0f}")

    sum_train = sum(r["train_pnl"] for r in rows)
    sum_test  = sum(r["test_pnl"]  for r in rows)
    sum_over  = sum(r["overfit_test_pnl"] for r in rows)
    sum_def   = sum(r["default_test_pnl"] for r in rows)
    print("  " + "-" * 60)
    # Per-day average is more interpretable
    print(f"  {'avg/d':>5}  "
          f"{sum_train / 6:>11.0f}  {sum_test / 3:>10.0f}  "
          f"{sum_over / 3:>15.0f}  {sum_def / 3:>12.0f}")

    print()
    print(f"  All-3-days tuned PnL (3 days, in-sample): 10,956 (= 3,652/day)")
    print(f"  Walk-forward OOS PnL (3 days):  {sum_test:.0f} (= {sum_test/3:.0f}/day)")
    print(f"  Default-params PnL   (3 days, untuned): {sum_def:.0f} (= {sum_def/3:.0f}/day)")
    print()
    if sum_test > 0 and sum_test < 10956:
        shrinkage = 1 - sum_test / 10956
        print(f"  → Estimated overfit shrinkage: {shrinkage*100:.0f}%.")
        print(f"    Live PnL expectation: ~{sum_test:.0f} ± noise on the same scale.")
    elif sum_test >= 10956:
        print(f"  → No shrinkage (OOS ≥ in-sample). Tuning is robust.")
    else:
        print(f"  → Tuning made OOS WORSE than default — hard overfit. "
              f"Revert to defaults (~{sum_def:.0f}).")


if __name__ == "__main__":
    main()
