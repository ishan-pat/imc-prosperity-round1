"""
Parameter sweep for Round 3 trader.

Single backtest ≈ 4.4s on this machine. With 10 cores → ~136 configs/min.
Plan that fits in 1 hour:
  - Phase 1: 1-D sensitivity per parameter (~80 runs, 1 min)
  - Phase 2: joint random search over a focused space (~2000 runs, ~15 min)
  - Phase 3: refinement around the best result (~200 runs, ~2 min)

Each subprocess imports the trader, monkey-patches its constants, runs the
backtest in silent mode, and returns (config, pnl).

Usage:
  python3 -m backtesting.round3_sweep --phase 1     # sensitivity only
  python3 -m backtesting.round3_sweep --phase 2     # full random search
  python3 -m backtesting.round3_sweep --phase all   # 1 → 2 → 3 (~20 min)
"""
import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Parameters we'll override; values pulled from trader.py defaults.
DEFAULT_PARAMS = {
    "GAMMA_HYDROGEL":         5e-7,
    "GAMMA_VELVETFRUIT":      2e-6,
    "SIGMA_HYDROGEL":         217.0,
    "SIGMA_VELVETFRUIT":      113.0,
    "K_HYDROGEL":             0.128,
    "K_VELVETFRUIT":          0.443,
    "MAX_HALF_SPREAD_HYDR":   8,
    "MAX_HALF_SPREAD_UND":    3,
    "MIN_HALF_SPREAD_HYDR":   1,
    "MIN_HALF_SPREAD_UND":    1,
    "MM_TOTAL_SIZE_HYDR":    120,
    "MM_TOTAL_SIZE_UND":      80,
    "MM_LEVELS":               4,
    "VOUCHER_MIN_EDGE":      2.0,
    "VOUCHER_EDGE_FRAC":     0.5,
    "VOUCHER_MAX_TRADE":      10,
    "SIGMA_EWMA_ALPHA":     0.005,
    "DELTA_DEAD_ZONE":      25.0,
}


def _run_one(params: dict) -> float:
    """Worker. Re-imports trader fresh, applies overrides, runs backtest."""
    # Force a fresh import per subprocess (safe because of fork start-method).
    import importlib
    import submissions.round3.trader as t
    importlib.reload(t)
    from backtesting.round3_backtest import run_backtest

    # Apply overrides — map our flat dict back to trader's module attrs.
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
    return run_backtest(t.Trader(), passive_share=0.5, verbose=False)


def _eval_grid(grid_list, pool, label):
    """Evaluate every config in grid_list in parallel; return [(cfg, pnl)]."""
    print(f"\n[{label}] {len(grid_list)} configs...")
    t0 = time.time()
    pnls = pool.map(_run_one, grid_list)
    dt = time.time() - t0
    out = list(zip(grid_list, pnls))
    print(f"[{label}] done in {dt:.1f}s ({dt / len(grid_list):.2f}s per run)")
    return out


# ───────────────────────────────────────────────────────────────────────────
# PHASE 1 — 1D sensitivity per parameter

PHASE1_GRIDS = {
    "GAMMA_HYDROGEL":      [1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5],
    "GAMMA_VELVETFRUIT":   [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5],
    "MAX_HALF_SPREAD_HYDR":[3, 4, 5, 6, 8, 10, 12, 15],
    "MAX_HALF_SPREAD_UND": [1, 2, 3, 4, 5, 6],
    "MIN_HALF_SPREAD_HYDR":[1, 2, 3],
    "MIN_HALF_SPREAD_UND": [1, 2],
    "MM_TOTAL_SIZE_HYDR":  [40, 80, 120, 160, 200],
    "MM_TOTAL_SIZE_UND":   [40, 80, 120, 160, 200],
    "MM_LEVELS":           [1, 2, 3, 4, 5, 6, 8],
    "VOUCHER_MIN_EDGE":    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
    "VOUCHER_EDGE_FRAC":   [0.0, 0.25, 0.5, 0.75, 1.0],
    "VOUCHER_MAX_TRADE":   [1, 3, 5, 10, 20, 50],
    "SIGMA_EWMA_ALPHA":    [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    "DELTA_DEAD_ZONE":     [5, 10, 15, 25, 40, 60, 100],
}


def phase1(pool):
    print("=" * 78)
    print("PHASE 1 — 1D sensitivity per parameter")
    print("=" * 78)
    all_results = {}
    best_per_param = {}
    for pname, values in PHASE1_GRIDS.items():
        configs = []
        for v in values:
            cfg = dict(DEFAULT_PARAMS)
            cfg[pname] = v
            configs.append(cfg)
        results = _eval_grid(configs, pool, f"phase1:{pname}")
        all_results[pname] = [(cfg[pname], pnl) for cfg, pnl in results]
        # Best value for this param
        best_v, best_p = max(all_results[pname], key=lambda x: x[1])
        baseline_p = next((p for v, p in all_results[pname] if v == DEFAULT_PARAMS[pname]), None)
        best_per_param[pname] = (best_v, best_p)
        improv = (best_p - baseline_p) if baseline_p is not None else 0
        print(f"  {pname:<24} default={DEFAULT_PARAMS[pname]:<10} "
              f"best={best_v:<10} pnl={best_p:>9.0f}  "
              f"(Δ from default={improv:>+8.0f})")
    return all_results, best_per_param


# ───────────────────────────────────────────────────────────────────────────
# PHASE 2 — Random search over a focused space (informed by Phase 1)

PHASE2_RANGES = {
    "GAMMA_HYDROGEL":      ("loguniform", 1e-7, 5e-6),
    "GAMMA_VELVETFRUIT":   ("loguniform", 5e-7, 5e-5),
    "MAX_HALF_SPREAD_HYDR":("intuniform", 3, 12),
    "MAX_HALF_SPREAD_UND": ("intuniform", 1, 6),
    "MM_TOTAL_SIZE_HYDR":  ("intuniform", 40, 200),
    "MM_TOTAL_SIZE_UND":   ("intuniform", 40, 160),
    "VOUCHER_MIN_EDGE":    ("uniform", 0.5, 5.0),
    "VOUCHER_MAX_TRADE":   ("intuniform", 3, 30),
    "SIGMA_EWMA_ALPHA":    ("loguniform", 0.0005, 0.05),
    "DELTA_DEAD_ZONE":     ("uniform", 5, 80),
}


def _sample_random(rng):
    cfg = dict(DEFAULT_PARAMS)
    for k, spec in PHASE2_RANGES.items():
        kind = spec[0]
        if kind == "loguniform":
            lo, hi = spec[1], spec[2]
            cfg[k] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        elif kind == "uniform":
            cfg[k] = rng.uniform(spec[1], spec[2])
        elif kind == "intuniform":
            cfg[k] = rng.randint(spec[1], spec[2])
    return cfg


def phase2(pool, n_samples=400, seed=42):
    print("=" * 78)
    print(f"PHASE 2 — Random search ({n_samples} samples)")
    print("=" * 78)
    rng = random.Random(seed)
    configs = [_sample_random(rng) for _ in range(n_samples)]
    results = _eval_grid(configs, pool, "phase2:random")
    results.sort(key=lambda x: -x[1])
    print(f"\nTop 10 of {n_samples}:")
    print(f"  {'PnL':>8}  {'γ_H':>8}  {'γ_V':>8}  {'mhs_H':>5}  {'mhs_U':>5}  "
          f"{'tot_H':>5}  {'tot_U':>5}  {'vEdg':>5}  {'vMax':>4}  "
          f"{'α':>7}  {'DDZ':>5}")
    for cfg, pnl in results[:10]:
        print(f"  {pnl:>8.0f}  "
              f"{cfg['GAMMA_HYDROGEL']:>8.1e}  {cfg['GAMMA_VELVETFRUIT']:>8.1e}  "
              f"{int(cfg['MAX_HALF_SPREAD_HYDR']):>5}  "
              f"{int(cfg['MAX_HALF_SPREAD_UND']):>5}  "
              f"{int(cfg['MM_TOTAL_SIZE_HYDR']):>5}  "
              f"{int(cfg['MM_TOTAL_SIZE_UND']):>5}  "
              f"{cfg['VOUCHER_MIN_EDGE']:>5.2f}  "
              f"{int(cfg['VOUCHER_MAX_TRADE']):>4}  "
              f"{cfg['SIGMA_EWMA_ALPHA']:>7.4f}  "
              f"{cfg['DELTA_DEAD_ZONE']:>5.1f}")
    return results


# ───────────────────────────────────────────────────────────────────────────
# PHASE 3 — Local refinement around best Phase 2 candidate

def phase3(pool, best_cfg, n_neighbors=80, seed=7):
    print("=" * 78)
    print(f"PHASE 3 — Local refinement ({n_neighbors} neighbors)")
    print("=" * 78)
    rng = random.Random(seed)

    def _jitter(cfg):
        c = dict(cfg)
        # Multiplicative jitter on log params, additive on linear
        c["GAMMA_HYDROGEL"]    *= math.exp(rng.uniform(-0.4, 0.4))
        c["GAMMA_VELVETFRUIT"] *= math.exp(rng.uniform(-0.4, 0.4))
        c["SIGMA_EWMA_ALPHA"]  *= math.exp(rng.uniform(-0.4, 0.4))
        c["MAX_HALF_SPREAD_HYDR"] = max(1, int(c["MAX_HALF_SPREAD_HYDR"]) + rng.randint(-2, 2))
        c["MAX_HALF_SPREAD_UND"]  = max(1, int(c["MAX_HALF_SPREAD_UND"])  + rng.randint(-1, 1))
        c["MM_TOTAL_SIZE_HYDR"]   = max(20, int(c["MM_TOTAL_SIZE_HYDR"])  + rng.randint(-30, 30))
        c["MM_TOTAL_SIZE_UND"]    = max(20, int(c["MM_TOTAL_SIZE_UND"])   + rng.randint(-30, 30))
        c["VOUCHER_MIN_EDGE"]     = max(0.1, c["VOUCHER_MIN_EDGE"] + rng.uniform(-0.5, 0.5))
        c["VOUCHER_MAX_TRADE"]    = max(1, int(c["VOUCHER_MAX_TRADE"]) + rng.randint(-5, 5))
        c["DELTA_DEAD_ZONE"]      = max(1, c["DELTA_DEAD_ZONE"] + rng.uniform(-15, 15))
        return c

    configs = [best_cfg] + [_jitter(best_cfg) for _ in range(n_neighbors)]
    results = _eval_grid(configs, pool, "phase3:refine")
    results.sort(key=lambda x: -x[1])
    print(f"\nTop 5 in neighborhood:")
    for cfg, pnl in results[:5]:
        print(f"  PnL={pnl:>8.0f}  γ_H={cfg['GAMMA_HYDROGEL']:.1e}  "
              f"mhs_H={int(cfg['MAX_HALF_SPREAD_HYDR'])}  "
              f"vEdg={cfg['VOUCHER_MIN_EDGE']:.2f}  "
              f"α={cfg['SIGMA_EWMA_ALPHA']:.4f}")
    return results


# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phase2-samples", type=int, default=400)
    parser.add_argument("--out", default=str(REPO_ROOT / "writeups" / "round3_sweep_results.json"))
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Baseline first
    print("Baseline (default params):")
    baseline_pnl = _run_one(DEFAULT_PARAMS)
    print(f"  PnL = {baseline_pnl:.0f}\n")

    summary = {"baseline_pnl": baseline_pnl, "phases": {}}

    with mp.Pool(args.workers) as pool:
        if args.phase in ("1", "all"):
            p1, best_per_param = phase1(pool)
            summary["phases"]["phase1"] = {
                pname: [{"value": v, "pnl": p} for v, p in vals]
                for pname, vals in p1.items()
            }
            summary["phases"]["phase1_best_per_param"] = {
                k: {"value": v[0], "pnl": v[1]} for k, v in best_per_param.items()
            }

        if args.phase in ("2", "all"):
            p2 = phase2(pool, n_samples=args.phase2_samples)
            summary["phases"]["phase2"] = [
                {"params": cfg, "pnl": pnl} for cfg, pnl in p2[:50]
            ]
            best_cfg, best_pnl = p2[0]

            if args.phase == "all":
                p3 = phase3(pool, best_cfg, n_neighbors=80)
                summary["phases"]["phase3"] = [
                    {"params": cfg, "pnl": pnl} for cfg, pnl in p3[:20]
                ]

    # Final recommendation
    print("\n" + "=" * 78)
    print("FINAL RECOMMENDATION")
    print("=" * 78)
    final = None
    if "phase3" in summary["phases"] and summary["phases"]["phase3"]:
        final = summary["phases"]["phase3"][0]
    elif "phase2" in summary["phases"] and summary["phases"]["phase2"]:
        final = summary["phases"]["phase2"][0]
    if final:
        print(f"  Best PnL found: {final['pnl']:.0f}")
        print(f"  vs baseline:    {final['pnl'] - baseline_pnl:+.0f}")
        print(f"\n  Apply these to submissions/round3/trader.py:\n")
        for k, v in final["params"].items():
            if isinstance(v, float):
                print(f"    {k:<24} = {v:.6g}")
            else:
                print(f"    {k:<24} = {v}")
    summary["final"] = final
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
