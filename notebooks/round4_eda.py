"""
Round 4 EDA — feeds the Trader's calibration constants.

Run as: python3 notebooks/round4_eda.py
Outputs: param block printed to stdout + writeups/round4_calibration.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

DATA = Path("data/round4")
DAYS = [1, 2, 3]
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
TTE_DAY_START = {1: 8, 2: 7, 3: 6}        # Round-3-style: TTE in DAYS, day 0 not in CSV
TICK = 100                                  # timestamp delta between snapshots
DAY_TS = 1_000_000

# ─────────────────────────────────────────────────────────────────────────────
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.concat([pd.read_csv(DATA/f"prices_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS], ignore_index=True)
    t = pd.concat([pd.read_csv(DATA/f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS], ignore_index=True)
    return p, t

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.1: Realized vol & autocorr for delta-1 products (calibrate AS σ)
def vol_and_autocorr(prices: pd.DataFrame, sym: str) -> dict:
    sub = prices[prices["product"] == sym].copy()
    out = {"sym": sym, "by_day": {}}
    log_returns_all = []
    for d in DAYS:
        s = sub[sub["day"] == d]["mid_price"].values
        lr = np.diff(np.log(s))
        log_returns_all.extend(lr.tolist())
        out["by_day"][d] = {
            "mid_mean": float(np.mean(s)),
            "mid_std": float(np.std(s)),
            "ret_std_per_tick": float(np.std(lr)),
            # per-day = per-tick × sqrt(ticks_per_day=10000)
            "sigma_per_day": float(np.std(lr) * np.sqrt(10_000)),
            # autocorr at lag 1 (mean-reversion / momentum diagnostic)
            "ac1": float(pd.Series(lr).autocorr(lag=1)),
            "ac5": float(pd.Series(lr).autocorr(lag=5)),
        }
    lr = np.array(log_returns_all)
    s_all = sub["mid_price"].values
    out["overall_sigma_per_day"] = float(np.std(lr) * np.sqrt(10_000))
    # Express σ in PRICE units (this is what AS uses)
    out["sigma_price_per_day"] = float(np.std(lr) * np.sqrt(10_000) * np.mean(s_all))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.3: Engle-Granger cointegration HYDROGEL ↔ VELVETFRUIT per day
def cointegration_test(prices: pd.DataFrame) -> dict:
    """Stage 1: regress y on x → residuals; Stage 2: ADF on residuals."""
    out = {}
    for d in DAYS:
        sub = prices[prices["day"] == d]
        h = sub[sub["product"] == "HYDROGEL_PACK"][["timestamp", "mid_price"]].rename(columns={"mid_price": "h"})
        v = sub[sub["product"] == "VELVETFRUIT_EXTRACT"][["timestamp", "mid_price"]].rename(columns={"mid_price": "v"})
        m = h.merge(v, on="timestamp")
        # OLS y = a + b·x
        x, y = m["v"].values, m["h"].values
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        # ADF approximation: Δr_t = ρ·r_{t-1} + ε; t-stat on ρ
        dr = np.diff(resid)
        rl = resid[:-1]
        rho = np.sum(dr * rl) / np.sum(rl ** 2)
        se = np.sqrt(np.sum((dr - rho * rl) ** 2) / (len(dr) - 1) / np.sum(rl ** 2))
        t_stat = rho / se
        out[d] = {
            "alpha": float(a), "beta": float(b),
            "resid_std": float(np.std(resid)),
            "rho": float(rho), "t_stat": float(t_stat),
            # Critical values for ADF (no constant) at 5% ≈ -1.95.  More negative = more stationary.
            "stationary_at_5pct": bool(t_stat < -2.86),  # 5% w/ constant
        }
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Step 2.x prereq: Aggression direction per Mark via Lee-Ready
def aggression_per_mark(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Lee-Ready: trade above prior mid → buyer-initiated; below → seller-initiated; at-mid → tick test."""
    # Build a per-(day, sym) timestamp→mid lookup
    midp = (prices[["day","timestamp","product","mid_price"]]
            .rename(columns={"product":"symbol"})
            .set_index(["day","symbol","timestamp"])["mid_price"])
    def prior_mid(day, sym, ts):
        # try exact, then ts-100
        try: return midp.loc[(day, sym, ts)]
        except KeyError: pass
        try: return midp.loc[(day, sym, ts - TICK)]
        except KeyError: return np.nan
    trades = trades.copy()
    trades["prior_mid"] = trades.apply(lambda r: prior_mid(r["day"], r["symbol"], r["timestamp"]), axis=1)
    trades = trades.dropna(subset=["prior_mid"])
    trades["initiator"] = np.where(
        trades["price"] > trades["prior_mid"], "buyer",
        np.where(trades["price"] < trades["prior_mid"], "seller", "midpoint"))
    # for each Mark, compute fraction of trades where they were the aggressor
    rows = []
    for mark in sorted(set(trades.buyer) | set(trades.seller)):
        legs_buy  = trades[trades.buyer  == mark]
        legs_sell = trades[trades.seller == mark]
        n = len(legs_buy) + len(legs_sell)
        if n == 0: continue
        agg = ((legs_buy.initiator == "buyer").sum() +
               (legs_sell.initiator == "seller").sum())
        rows.append({"mark": mark, "n_legs": n,
                     "aggressor_pct": float(agg / n),
                     "passive_pct": float(((legs_buy.initiator == "seller").sum() +
                                            (legs_sell.initiator == "buyer").sum()) / n),
                     "midpoint_pct": float(((legs_buy.initiator == "midpoint").sum() +
                                             (legs_sell.initiator == "midpoint").sum()) / n)})
    return pd.DataFrame(rows).sort_values("aggressor_pct", ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2.x: post-fill price drift after Mark 14 trades — toxicity decay
def toxicity_decay(prices: pd.DataFrame, trades: pd.DataFrame, mark: str = "Mark 14") -> dict:
    midp = (prices[["day","timestamp","product","mid_price"]]
            .rename(columns={"product":"symbol"})
            .set_index(["day","symbol","timestamp"])["mid_price"])
    sel = trades[(trades.buyer == mark) | (trades.seller == mark)].copy()
    sel["sign"] = np.where(sel.buyer == mark, +1, -1)  # +1 if mark bought
    decays = {}
    for h in (1, 5, 10, 30, 100, 300, 1000):
        deltas = []
        for _, r in sel.iterrows():
            try:
                m_now = midp.loc[(r["day"], r["symbol"], r["timestamp"])]
                m_fwd = midp.loc[(r["day"], r["symbol"], r["timestamp"] + h*TICK)]
            except KeyError: continue
            deltas.append(r["sign"] * (m_fwd - m_now))   # positive = drift *with* mark
        if deltas:
            decays[h] = {"mean": float(np.mean(deltas)), "n": len(deltas)}
    return decays

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.4: BS, IV inversion, raw-SVI fit per day
def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def implied_vol(price, S, K, T):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or T <= 0: return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, s) - price, 1e-4, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan

def fit_svi(k, w):
    """Raw SVI: w(k) = a + b·(ρ(k-m) + sqrt((k-m)² + σ²))"""
    def resid(p, k=k, w=w):
        a, b, rho, m, sg = p
        return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sg**2)) - w
    p0 = [0.0, 0.05, 0.0, 0.0, 0.05]
    bounds = ([-1, 0,    -0.999, -1, 1e-4], [1, 5, 0.999, 1, 1])
    sol = least_squares(resid, p0, bounds=bounds)
    return dict(zip(["a","b","rho","m","sigma"], sol.x.tolist())), float(np.sqrt(np.mean(sol.fun**2)))

def voucher_iv_and_svi(prices: pd.DataFrame) -> dict:
    """Per day: collect (K, mid_voucher, S=mid_VEV) at every timestamp, invert IV, fit SVI."""
    out = {}
    for d in DAYS:
        T = TTE_DAY_START[d]
        sub = prices[prices["day"] == d]
        S_ts = (sub[sub["product"] == "VELVETFRUIT_EXTRACT"]
                  .set_index("timestamp")["mid_price"])
        ks, ws, sigmas = [], [], []
        per_strike = {}
        for K in STRIKES:
            sym = f"VEV_{K}"
            v = (sub[sub["product"] == sym]
                   .set_index("timestamp")["mid_price"])
            df = pd.DataFrame({"S": S_ts, "V": v}).dropna()
            if df.empty: continue
            ivs = [implied_vol(p, S, K, T) for p, S in zip(df["V"], df["S"])]
            ivs = [v for v in ivs if v is not None and not np.isnan(v)]
            if not ivs: continue
            sigma_K = float(np.median(ivs))
            S_mean = float(df["S"].mean())
            per_strike[K] = {"sigma": sigma_K, "n": len(ivs), "S_mean": S_mean}
            # for SVI: k = ln(K/S), w = sigma² · T
            ks.append(np.log(K / S_mean))
            ws.append(sigma_K**2 * T)
            sigmas.append(sigma_K)
        if ks:
            params, rmse = fit_svi(np.array(ks), np.array(ws))
            out[d] = {"per_strike": per_strike, "svi": params, "svi_rmse": rmse}
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Active voucher universe: which strikes have enough trade activity?
def voucher_activity(trades: pd.DataFrame) -> pd.DataFrame:
    voucher_trades = trades[trades.symbol.str.startswith("VEV_")]
    counts = voucher_trades.groupby("symbol").agg(n_trades=("symbol","count"),
                                                  total_qty=("quantity","sum"),
                                                  mean_price=("price","mean")).sort_values("n_trades", ascending=False)
    return counts

# ─────────────────────────────────────────────────────────────────────────────
def main():
    prices, trades = load()

    print("="*78)
    print("STEP 3.1 — Realized vol & autocorr (calibrates AS σ)")
    print("="*78)
    for sym in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
        v = vol_and_autocorr(prices, sym)
        print(f"\n{sym}: σ_per_day_log = {v['overall_sigma_per_day']:.5f}, σ_price_per_day = {v['sigma_price_per_day']:.2f}")
        for d, s in v["by_day"].items():
            print(f"  day {d}: mid={s['mid_mean']:.0f}  σ_log={s['sigma_per_day']:.5f}  AC1={s['ac1']:+.3f}  AC5={s['ac5']:+.3f}")

    print("\n" + "="*78)
    print("STEP 3.3 — Engle-Granger cointegration HYDROGEL ↔ VELVETFRUIT")
    print("="*78)
    coint = cointegration_test(prices)
    for d, r in coint.items():
        flag = "STATIONARY (cointegrated)" if r["stationary_at_5pct"] else "non-stationary (NOT cointegrated)"
        print(f"  day {d}: β={r['beta']:+.3f}  resid_std={r['resid_std']:.2f}  ADF t={r['t_stat']:+.2f}  → {flag}")

    print("\n" + "="*78)
    print("STEP 2 prereq — Aggression direction per Mark (Lee-Ready)")
    print("="*78)
    agg = aggression_per_mark(prices, trades)
    print(agg.to_string(index=False))

    print("\n" + "="*78)
    print("STEP 2 prereq — Mark 14 toxicity decay (price drift after Mark 14 fills)")
    print("="*78)
    print("Aggregated across all symbols.  Positive = price drifts WITH Mark 14 (i.e., they were right).")
    decay = toxicity_decay(prices, trades, "Mark 14")
    for h, r in decay.items():
        print(f"  +{h:>4} ticks: drift = {r['mean']:+.3f}  (n={r['n']})")

    print("\n" + "="*78)
    print("STEP 3.4 — Voucher IV per strike per day + raw-SVI fit")
    print("="*78)
    iv = voucher_iv_and_svi(prices)
    for d, r in iv.items():
        T = TTE_DAY_START[d]
        print(f"\nday {d} (TTE={T}):")
        print(f"  SVI: a={r['svi']['a']:.6f}  b={r['svi']['b']:.5f}  ρ={r['svi']['rho']:+.3f}  "
              f"m={r['svi']['m']:+.4f}  σ={r['svi']['sigma']:.4f}  RMSE_w={r['svi_rmse']:.6f}")
        for K, ps in r["per_strike"].items():
            print(f"    K={K}: σ={ps['sigma']:.4f}  n={ps['n']}  (S~{ps['S_mean']:.0f})")

    print("\n" + "="*78)
    print("STEP 3 prereq — voucher activity (drives ACTIVE_STRIKES universe)")
    print("="*78)
    print(voucher_activity(trades).to_string())

    # Pack calibration to JSON for the trader
    out = {
        "vol": {sym: vol_and_autocorr(prices, sym) for sym in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]},
        "cointegration": coint,
        "voucher_iv_svi": iv,
    }
    Path("writeups").mkdir(exist_ok=True)
    Path("writeups/round4_calibration.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved → writeups/round4_calibration.json")

if __name__ == "__main__":
    main()
