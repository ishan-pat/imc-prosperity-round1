"""
IMC Prosperity Round 4 manual: AETHER_CRYSTAL one-shot portfolio.

Dynamics (fully specified, do NOT recalibrate):
    dS/S = sigma dW   under risk-neutral measure (mu = 0)
    sigma = 2.51, S0 = 50, 252 days/yr, 4 steps/day, dt = 1/1008
    Discrete barrier monitoring on the 4-steps/day grid.

Pricing references (cited inline below):
    Black & Scholes (1973)            -- vanilla closed-form
    Merton (1973)                     -- put-call parity bounds
    Rubinstein (1991)                 -- chooser decomposition
    Rubinstein & Reiner (1991)        -- binary / digital
    Broadie, Glasserman, Kou (1997)   -- discrete-barrier continuity correction
    Boyle (1977)                      -- Monte Carlo for option pricing
    Glasserman (2003)                 -- antithetic VR

Run end-to-end: python manual/round4_aether_crystal.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------------------
# Specified GBM dynamics
# --------------------------------------------------------------------------------------
S0              = 50.0
SIGMA           = 2.51                    # annualised; ~251%
MU              = 0.0
STEPS_PER_DAY   = 4
DAYS_PER_YEAR   = 252
DT              = 1.0 / (DAYS_PER_YEAR * STEPS_PER_DAY)        # 1/1008
# Wiki: "2 weeks" = 10 trading days, "3 weeks" = 15 trading days.
# UI labels "T+14" / "T+21" are CALENDAR days, not trading days.
T_14            = 10 / DAYS_PER_YEAR                            # "2 weeks" = 10 td
T_21            = 15 / DAYS_PER_YEAR                            # "3 weeks" = 15 td
N_STEPS_14      = 10 * STEPS_PER_DAY                            # 40
N_STEPS_21      = 15 * STEPS_PER_DAY                            # 60
CONTRACT_MULT   = 3000                                           # PnL multiplier per contract

# --------------------------------------------------------------------------------------
# Closed-form pricers (Black & Scholes 1973; Rubinstein & Reiner 1991; Rubinstein 1991)
# r = q = 0 so all discount factors are 1.
# --------------------------------------------------------------------------------------
def bs_call(S, K, T, sigma):
    if T <= 0: return max(S - K, 0.0)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / sT
    d2 = d1 - sT
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put(S, K, T, sigma):
    # Merton (1973) put-call parity: P = C - S + K (with r=q=0)
    return bs_call(S, K, T, sigma) - S + K

def bin_put(S, K, T, sigma, payoff=10.0):
    # Rubinstein & Reiner (1991): cash-or-nothing put
    if T <= 0: return payoff if S < K else 0.0
    sT = sigma * np.sqrt(T)
    d2 = (np.log(S / K) - 0.5 * sigma**2 * T) / sT
    return payoff * norm.cdf(-d2)

def chooser_cf(S, K, t_choose, T_expire, sigma):
    # Rubinstein (1991) "Options for the Undecided": with q=0,
    # chooser = C(S,K,T) + P(S, K, t_choose).  Rational-choice value.
    return bs_call(S, K, T_expire, sigma) + bs_put(S, K, t_choose, sigma)

# --------------------------------------------------------------------------------------
# Path simulator (Boyle 1977; antithetic per Glasserman 2003 ch.4)
# --------------------------------------------------------------------------------------
def simulate_paths(n_paths, n_steps, S0=S0, sigma=SIGMA, dt=DT, seed=0, antithetic=True):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        Z_h  = rng.standard_normal((half, n_steps))
        Z    = np.concatenate([Z_h, -Z_h], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    incr      = (-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_path  = np.cumsum(incr, axis=1)
    paths     = S0 * np.exp(log_path)
    return np.concatenate([np.full((paths.shape[0], 1), S0), paths], axis=1)   # (n_paths, n_steps+1)

# --------------------------------------------------------------------------------------
# MC pricers for path-dependent legs
# --------------------------------------------------------------------------------------
def ko_put_price_mc(K=45, B=35, n_steps=N_STEPS_21, n_paths=400_000, seed=42, sigma=SIGMA):
    paths   = simulate_paths(n_paths, n_steps, sigma=sigma, seed=seed)
    knocked = (paths[:, 1:] <= B).any(axis=1)         # discrete monitoring at each step >0
    S_T     = paths[:, -1]
    payoff  = np.where(knocked, 0.0, np.maximum(K - S_T, 0.0))
    return payoff.mean(), payoff.std() / np.sqrt(n_paths)

def chooser_autoconvert_price_mc(K=50, t_choose=N_STEPS_14, T_expire=N_STEPS_21,
                                 n_paths=400_000, seed=43, sigma=SIGMA):
    """Auto-convert at t_choose: pick call if S_t >= K else put.  This is the spec."""
    paths    = simulate_paths(n_paths, T_expire, sigma=sigma, seed=seed)
    S_choose = paths[:, t_choose]
    S_T      = paths[:, -1]
    is_call  = S_choose >= K
    payoff   = np.where(is_call, np.maximum(S_T - K, 0.0), np.maximum(K - S_T, 0.0))
    return payoff.mean(), payoff.std() / np.sqrt(n_paths)

# --------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------
@dataclass
class Inst:
    name: str
    typ:  str           # call/put/binary_put/chooser/ko_put/underlying
    K:    float | None
    T_steps: int | None
    bid:  float
    ask:  float
    size: int
    extra: dict | None = None

INSTRUMENTS = [
    Inst("AC",         "underlying", None, None, 49.975, 50.025, 200),
    Inst("AC_50_P",    "put",   50, N_STEPS_21, 12.00, 12.05, 50),
    Inst("AC_50_C",    "call",  50, N_STEPS_21, 12.00, 12.05, 50),
    Inst("AC_35_P",    "put",   35, N_STEPS_21,  4.33,  4.35, 50),
    Inst("AC_40_P",    "put",   40, N_STEPS_21,  6.50,  6.55, 50),
    Inst("AC_45_P",    "put",   45, N_STEPS_21,  9.05,  9.10, 50),
    Inst("AC_60_C",    "call",  60, N_STEPS_21,  8.80,  8.85, 50),
    Inst("AC_50_P_2",  "put",   50, N_STEPS_14,  9.70,  9.75, 50),
    Inst("AC_50_C_2",  "call",  50, N_STEPS_14,  9.70,  9.75, 50),
    Inst("AC_50_CO",   "chooser", 50, N_STEPS_21, 22.20, 22.30, 50,
         {"t_choose": N_STEPS_14}),
    Inst("AC_40_BP",   "binary_put", 40, N_STEPS_21, 5.00, 5.10, 50,
         {"payoff": 10.0}),
    Inst("AC_45_KO",   "ko_put", 45, N_STEPS_21, 0.15, 0.175, 500,
         {"B": 35.0}),
]
NAME = {i.name: i for i in INSTRUMENTS}

# --------------------------------------------------------------------------------------
# Step 1 + 2: print fair value table and parity diagnostics
# --------------------------------------------------------------------------------------
def step1_fair_table():
    fairs: dict[str, float] = {}
    fairs["AC"]        = S0
    fairs["AC_50_P"]   = bs_put (S0, 50, T_21, SIGMA)
    fairs["AC_50_C"]   = bs_call(S0, 50, T_21, SIGMA)
    fairs["AC_35_P"]   = bs_put (S0, 35, T_21, SIGMA)
    fairs["AC_40_P"]   = bs_put (S0, 40, T_21, SIGMA)
    fairs["AC_45_P"]   = bs_put (S0, 45, T_21, SIGMA)
    fairs["AC_60_C"]   = bs_call(S0, 60, T_21, SIGMA)
    fairs["AC_50_P_2"] = bs_put (S0, 50, T_14, SIGMA)
    fairs["AC_50_C_2"] = bs_call(S0, 50, T_14, SIGMA)
    fairs["AC_50_CO"]  = chooser_cf(S0, 50, T_14, T_21, SIGMA)
    fairs["AC_40_BP"]  = bin_put(S0, 40, T_21, SIGMA, payoff=10.0)

    ko_fair, ko_se     = ko_put_price_mc(n_paths=400_000, seed=42)
    fairs["AC_45_KO"]  = ko_fair

    # Cross-check chooser via MC (auto-convert spec) -- should be ~ rational chooser
    ch_mc, ch_se = chooser_autoconvert_price_mc(n_paths=400_000, seed=43)

    print("\n=== STEP 1: Fair values vs market ===")
    print(f"{'Inst':12s} {'Fair':>9s} {'Mid':>9s} {'Bid':>9s} {'Ask':>9s} "
          f"{'EdgeBuy':>9s} {'EdgeSell':>9s} {'Buy%':>7s}")
    for ins in INSTRUMENTS:
        f   = fairs[ins.name]
        mid = 0.5 * (ins.bid + ins.ask)
        eb  = f - ins.ask                        # edge if we buy at ask
        es  = ins.bid - f                        # edge if we sell at bid
        bp  = 100.0 * eb / max(ins.ask, 1e-9)
        print(f"{ins.name:12s} {f:9.4f} {mid:9.4f} {ins.bid:9.3f} {ins.ask:9.3f} "
              f"{eb:+9.4f} {es:+9.4f} {bp:+7.2f}")

    print(f"\nKO MC SE = {ko_se:.4f}   Chooser MC (auto-convert) = "
          f"{ch_mc:.4f} +- {ch_se:.4f}  (rational closed-form: {fairs['AC_50_CO']:.4f})")

    # Parity / consistency block (Merton 1973; Rubinstein 1991)
    print("\n=== STEP 2: Internal consistency ===")
    pcp_21 = (NAME["AC_50_C"].bid + NAME["AC_50_C"].ask)/2 - (NAME["AC_50_P"].bid + NAME["AC_50_P"].ask)/2
    pcp_14 = (NAME["AC_50_C_2"].bid + NAME["AC_50_C_2"].ask)/2 - (NAME["AC_50_P_2"].bid + NAME["AC_50_P_2"].ask)/2
    print(f"Put-call parity (mid): C50-P50 (21d) = {pcp_21:+.4f}   (14d) = {pcp_14:+.4f}   "
          f"(should be 0 with r=q=0, S=K)")

    # Implied vol of ATM 21d / 14d (root-find via series for ATM with r=q=0)
    def iv_atm(C, T):
        # ATM, r=q=0:  C = S * (2 N(sigma sqrt(T)/2) - 1).  Invert.
        x = (C / S0 + 1) / 2
        return 2 * norm.ppf(x) / np.sqrt(T)
    iv21 = iv_atm(0.5*(NAME["AC_50_C"].bid + NAME["AC_50_C"].ask), T_21)
    iv14 = iv_atm(0.5*(NAME["AC_50_C_2"].bid + NAME["AC_50_C_2"].ask), T_14)
    print(f"Implied vols at mid: ATM 21d = {iv21:.3f},  ATM 14d = {iv14:.3f}   (true sigma = {SIGMA:.3f})")

    # Chooser identity: chooser_market vs C_50_21 + P_50_14 (the synthetic that replicates)
    syn_mid  = 0.5*(NAME["AC_50_C"].bid + NAME["AC_50_C"].ask) + 0.5*(NAME["AC_50_P_2"].bid + NAME["AC_50_P_2"].ask)
    ch_mid   = 0.5*(NAME["AC_50_CO"].bid + NAME["AC_50_CO"].ask)
    print(f"Chooser identity (mid): chooser={ch_mid:.4f}, C50_21+P50_14={syn_mid:.4f}, "
          f"diff={ch_mid-syn_mid:+.4f}  (chooser RICH by this amount)")

    # Executable arb: sell chooser bid, buy synthetic at asks
    arb_locked = NAME["AC_50_CO"].bid - (NAME["AC_50_C"].ask + NAME["AC_50_P_2"].ask)
    print(f"Static arb (sell chooser / buy synthetic at execution): {arb_locked:+.4f} per contract  "
          f"(>=0 means risk-free if rational chooser holds)")

    return fairs

# --------------------------------------------------------------------------------------
# Step 4: portfolio Monte-Carlo evaluator
# --------------------------------------------------------------------------------------
def leg_payoff_per_path(inst: Inst, paths_21: np.ndarray) -> np.ndarray:
    """Terminal payoff (in option points, not yet * size or multiplier) for each path.

    paths_21 has shape (n_paths, N_STEPS_21+1) with column 0 = S0.
    """
    S_T_21  = paths_21[:, N_STEPS_21]
    if inst.typ == "underlying":
        # holding 1 share: PnL at expiry equals S_T - entry.  Use 21d horizon.
        return S_T_21
    if inst.typ == "call":
        S_T = paths_21[:, inst.T_steps]
        return np.maximum(S_T - inst.K, 0.0)
    if inst.typ == "put":
        S_T = paths_21[:, inst.T_steps]
        return np.maximum(inst.K - S_T, 0.0)
    if inst.typ == "binary_put":
        S_T = paths_21[:, inst.T_steps]
        return np.where(S_T < inst.K, inst.extra["payoff"], 0.0)
    if inst.typ == "chooser":
        # Auto-convert at t_choose: ITM rule per spec.
        t_c       = inst.extra["t_choose"]
        S_choose  = paths_21[:, t_c]
        S_T       = paths_21[:, inst.T_steps]
        is_call   = S_choose >= inst.K
        return np.where(is_call, np.maximum(S_T - inst.K, 0.0),
                                  np.maximum(inst.K - S_T, 0.0))
    if inst.typ == "ko_put":
        B       = inst.extra["B"]
        # discrete monitoring: every grid point >0 up to T_steps
        knocked = (paths_21[:, 1:inst.T_steps+1] <= B).any(axis=1)
        S_T     = paths_21[:, inst.T_steps]
        return np.where(knocked, 0.0, np.maximum(inst.K - S_T, 0.0))
    raise ValueError(inst.typ)

def portfolio_pnl(positions: dict[str, int], paths_21: np.ndarray):
    """Returns per-path PnL vector in DOLLARS (already x CONTRACT_MULT).

    positions[name] = signed integer (positive = bought at ask, negative = sold at bid).
    """
    n_paths = paths_21.shape[0]
    pnl = np.zeros(n_paths)
    for name, qty in positions.items():
        if qty == 0: continue
        ins = NAME[name]
        leg = leg_payoff_per_path(ins, paths_21)
        if qty > 0:
            entry = ins.ask                                # we paid ask
        else:
            entry = ins.bid                                # we received bid
        # For underlying, "entry" is just the mid-of-book price we paid/received
        if ins.typ == "underlying":
            # multiplier 1 per "share size" -- but per problem statement contract_mult=3000 universally
            mult = CONTRACT_MULT
        else:
            mult = CONTRACT_MULT
        pnl_per_unit = (leg - entry) if qty > 0 else (entry - leg)
        pnl += abs(qty) * mult * pnl_per_unit
    return pnl

# --------------------------------------------------------------------------------------
# Step 3: candidate portfolios
# --------------------------------------------------------------------------------------
def make_max_ev(fairs: dict[str, float]) -> dict[str, int]:
    """Max-EV: buy at max size whenever fair > ask, sell at max size whenever bid > fair.
    KO put is rich (MC fair ~0.13 vs ask 0.175), so this skips it on the buy side.
    """
    pos: dict[str, int] = {}
    for ins in INSTRUMENTS:
        f = fairs[ins.name]
        if ins.typ == "underlying":
            pos[ins.name] = 0
            continue
        if f > ins.ask:
            pos[ins.name] = +ins.size
        elif ins.bid > f:
            pos[ins.name] = -ins.size                  # short instruments richer than fair
        else:
            pos[ins.name] = 0
    return pos

def make_risk_adjusted(fairs: dict[str, float]) -> dict[str, int]:
    """Half-size every leg of the clean-arb portfolio to cut variance further."""
    pos = make_arb_clean(fairs)
    return {k: (v // 2 if v % 2 == 0 else (v + 1) // 2 * (1 if v > 0 else -1))
            for k, v in pos.items()}

def make_arb_only(fairs: dict[str, float]) -> dict[str, int]:
    """Just the chooser arb, nothing else.  Lowest-risk +EV portfolio."""
    pos = {ins.name: 0 for ins in INSTRUMENTS}
    pos["AC_50_CO"]   = -50
    pos["AC_50_C"]    = +50
    pos["AC_50_P_2"]  = +50
    return pos

def make_arb_plus_high_sharpe(fairs: dict[str, float]) -> dict[str, int]:
    """Chooser arb + the highest-Sharpe bounded legs (BP short + KO long).
    Skips AC_50_C_2 (low Sharpe contribution: +$18k mean for ~$3M std)."""
    pos = make_arb_only(fairs)
    pos["AC_40_BP"] = -50
    pos["AC_45_KO"] = +500
    return pos

def make_robust(fairs: dict[str, float]) -> dict[str, int]:
    """
    Robust: drop ATM straddles entirely (highest path-near-50 loss concentration);
    keep bounded-payoff puts and the chooser (still has the largest absolute edge).
    """
    pos = {ins.name: 0 for ins in INSTRUMENTS}
    pos["AC_50_CO"]  = 50
    pos["AC_35_P"]   = 50
    pos["AC_40_P"]   = 50
    pos["AC_45_P"]   = 50
    pos["AC_60_C"]   = 25                              # half-size the unbounded leg
    pos["AC_40_BP"]  = 50                              # bounded; tiny but positive edge
    return pos

def make_short_ko(fairs: dict[str, float]) -> dict[str, int]:
    """Like risk-adjusted but also short the rich KO at full size."""
    pos = make_risk_adjusted(fairs)
    if NAME["AC_45_KO"].bid > fairs["AC_45_KO"]:
        pos["AC_45_KO"] = -500
    return pos

def make_arb_aware(fairs: dict[str, float]) -> dict[str, int]:
    """
    Force the chooser arb structure (sell chooser, buy 21d call, buy 14d put)
    even when individual legs have small negative edge.  By Rubinstein (1991)
    identity: chooser = C(K, T_3w) + P(K, T_2w).  At any S=K with q=r=0 the
    arb is +EV with bounded variance (only path-risk is S_3w - S_2w in the
    down-regime, mean zero).  Then layer on remaining individual edges.
    """
    pos = {ins.name: 0 for ins in INSTRUMENTS}
    # Chooser arb leg
    pos["AC_50_CO"]   = -50
    pos["AC_50_C"]    = +50
    pos["AC_50_P_2"]  = +50
    # Other independent edges per fair-value table
    for ins in INSTRUMENTS:
        if ins.name in {"AC_50_CO", "AC_50_C", "AC_50_P_2", "AC"}:
            continue
        f = fairs[ins.name]
        if f > ins.ask:
            pos[ins.name] = +ins.size
        elif ins.bid > f:
            pos[ins.name] = -ins.size
    return pos

def make_arb_clean(fairs: dict[str, float]) -> dict[str, int]:
    """
    Cleanest portfolio: chooser arb + only the bounded-loss positive-edge legs.
    Drop AC_60_C short (only +0.008 edge vs unbounded upside loss); keep
    everything whose worst-case loss is bounded.
    """
    pos = {ins.name: 0 for ins in INSTRUMENTS}
    # Chooser arb (sell chooser + buy synthetic).  Per-path PnL is +0.40 in
    # the up regime and 0.40 + (S_3w - S_2w) in the down regime (mean zero).
    pos["AC_50_CO"]   = -50
    pos["AC_50_C"]    = +50
    pos["AC_50_P_2"]  = +50
    # 2w ATM call: same vol-cheapness as P_2w, fully separate contract.  Loss
    # bounded by premium 9.75/unit.
    pos["AC_50_C_2"]  = +50
    # Binary put short: payoff ∈ {0, 10}, so PnL ∈ [-5, +5] per unit. Bounded.
    pos["AC_40_BP"]   = -50
    # KO put long: loss bounded by premium 0.175/unit.
    pos["AC_45_KO"]   = +500
    return pos

# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
def diag_portfolio(label, positions, paths_21, paths_21_lo, paths_21_hi, n_boot=10_000, sample=100):
    pnl = portfolio_pnl(positions, paths_21)
    mu, sd = pnl.mean(), pnl.std()
    q01, q05, q50, q95 = np.quantile(pnl, [0.01, 0.05, 0.5, 0.95])
    p_loss = (pnl < 0).mean()
    sharpe = mu / sd if sd > 0 else float('nan')

    # 100-sample bootstrap (this is the platform's actual scoring distribution)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(pnl), size=(n_boot, sample))
    boot = pnl[idx].mean(axis=1)
    bq = np.quantile(boot, [0.01, 0.05, 0.5, 0.95, 0.99])

    # Vol stress
    pnl_lo = portfolio_pnl(positions, paths_21_lo).mean()
    pnl_hi = portfolio_pnl(positions, paths_21_hi).mean()

    # Per-leg variance contribution (cov-with-portfolio / var(portfolio))
    contribs = []
    for name, qty in positions.items():
        if qty == 0: continue
        leg_only = portfolio_pnl({name: qty}, paths_21)
        contribs.append((name, qty, leg_only.mean(), leg_only.std(),
                         np.cov(leg_only, pnl)[0,1]/sd**2 if sd>0 else 0))
    contribs.sort(key=lambda x: -abs(x[4]))

    print(f"\n=== {label} ===")
    print(f"  positions (nonzero): " + ", ".join(f"{k}:{v:+d}" for k,v in positions.items() if v))
    print(f"  Mean PnL              ${mu:>14,.0f}")
    print(f"  Median PnL            ${q50:>14,.0f}")
    print(f"  Std                   ${sd:>14,.0f}")
    print(f"  Sharpe-like (mu/sd)    {sharpe:>14.3f}")
    print(f"  P(loss)                {p_loss:>14.2%}")
    print(f"  5th pct                ${q05:>14,.0f}")
    print(f"  1st pct                ${q01:>14,.0f}")
    print(f"  Worst single sim       ${pnl.min():>14,.0f}")
    print(f"  100-sample bootstrap (platform scoring distribution):")
    print(f"     1%/5%/50%/95%/99%   ${bq[0]:>11,.0f} / ${bq[1]:>11,.0f} / ${bq[2]:>11,.0f} / "
          f"${bq[3]:>11,.0f} / ${bq[4]:>11,.0f}")
    print(f"  Vol stress: sigma=2.0 mean ${pnl_lo:>12,.0f}    sigma=3.0 mean ${pnl_hi:>12,.0f}")
    print(f"  Top variance contributors (leg, qty, leg_mean, leg_std, var_share):")
    for n, q, m, s, vs in contribs[:6]:
        print(f"     {n:11s}  qty={q:+4d}  mean=${m:>12,.0f}  std=${s:>12,.0f}  varshare={vs:+.2%}")

    return {"mean": mu, "std": sd, "p01": q01, "p05": q05, "boot": bq,
            "stress_lo": pnl_lo, "stress_hi": pnl_hi, "pnl": pnl}

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    np.set_printoptions(suppress=True, linewidth=140)

    print("="*92)
    print(f"AETHER_CRYSTAL: S0={S0}, sigma={SIGMA}, r=mu=0, dt=1/{int(1/DT)}, T_21=21d, T_14=14d")
    print("="*92)

    fairs = step1_fair_table()

    # Big shared sim used by all portfolios (Step 4)
    print("\nSimulating 200,000 paths to T=21d (antithetic) ...")
    paths_21    = simulate_paths(200_000, N_STEPS_21, seed=2026)
    paths_21_lo = simulate_paths(200_000, N_STEPS_21, sigma=2.0, seed=2026)
    paths_21_hi = simulate_paths(200_000, N_STEPS_21, sigma=3.0, seed=2026)

    print("\n=== STEP 3+4: candidate portfolios ===")
    pf_max   = make_max_ev(fairs)
    pf_arb   = make_arb_aware(fairs)
    pf_clean = make_arb_clean(fairs)
    pf_half  = make_risk_adjusted(fairs)
    pf_only  = make_arb_only(fairs)
    pf_high  = make_arb_plus_high_sharpe(fairs)

    r_max   = diag_portfolio("PORTFOLIO A: PURE EV-MAX (per-leg, naked shorts)", pf_max,   paths_21, paths_21_lo, paths_21_hi)
    r_arb   = diag_portfolio("PORTFOLIO E: CHOOSER-ARB + per-leg",               pf_arb,   paths_21, paths_21_lo, paths_21_hi)
    r_clean = diag_portfolio("PORTFOLIO F: CHOOSER-ARB + BOUNDED LEGS ONLY",     pf_clean, paths_21, paths_21_lo, paths_21_hi)
    r_half  = diag_portfolio("PORTFOLIO H: F at half size",                      pf_half,  paths_21, paths_21_lo, paths_21_hi)
    r_only  = diag_portfolio("PORTFOLIO G: CHOOSER ARB ONLY (lowest risk)",      pf_only,  paths_21, paths_21_lo, paths_21_hi)
    r_high  = diag_portfolio("PORTFOLIO I: ARB + BP_short + KO_long (best Sharpe)", pf_high, paths_21, paths_21_lo, paths_21_hi)

    # Step 6 ranking
    print("\n=== STEP 6: comparison ===")
    print(f"{'Portfolio':40s} {'mean':>14s} {'5%boot':>14s} {'1%boot':>14s} {'worst sim':>16s}")
    for label, r in [("A per-leg, naked shorts (DANGEROUS)", r_max),
                     ("E chooser-arb + per-leg",             r_arb),
                     ("F chooser-arb + bounded legs only",   r_clean),
                     ("I arb + BP_short + KO_long (no C_2w)", r_high),
                     ("H = F at half size",                  r_half),
                     ("G chooser arb ONLY",                  r_only)]:
        print(f"{label:40s} ${r['mean']:>12,.0f} ${r['boot'][1]:>12,.0f} ${r['boot'][0]:>12,.0f} "
              f"${r['pnl'].min():>14,.0f}")

    # Step 7: pick the portfolio that maximises the 5%-bootstrap floor among those whose
    # mean PnL is at least 70% of the best mean.  This trades a little EV for tail safety.
    candidates = {"A_max": (pf_max, r_max), "E_arb": (pf_arb, r_arb),
                  "F_clean": (pf_clean, r_clean), "H_half": (pf_half, r_half),
                  "G_only": (pf_only, r_only), "I_high": (pf_high, r_high)}
    # Decision rule: maximize 5%-bootstrap floor among portfolios with mean >= 0.7 * best mean
    best_mean = max(r["mean"] for _, r in candidates.values())
    eligible  = {k: v for k,v in candidates.items() if v[1]["mean"] >= 0.7 * best_mean}
    chosen_k  = max(eligible, key=lambda k: eligible[k][1]["boot"][1])
    chosen_pf = eligible[chosen_k][0]

    print(f"\n=== STEP 7: ORDER TICKET (auto-selected: {chosen_k}) ===")
    print(f"{'Instrument':12s} {'Side':6s} {'Qty':>5s} {'Px':>8s} {'Notional':>14s}")
    notional = 0.0
    for name, qty in chosen_pf.items():
        if qty == 0: continue
        ins  = NAME[name]
        side = "BUY" if qty > 0 else "SELL"
        px   = ins.ask if qty > 0 else ins.bid
        n    = abs(qty) * px * CONTRACT_MULT * (1 if qty > 0 else -1)
        notional += n
        print(f"{name:12s} {side:6s} {abs(qty):>5d} {px:>8.3f} ${n:>13,.0f}")
    print(f"  Net premium paid: ${notional:,.0f}")

if __name__ == "__main__":
    main()
