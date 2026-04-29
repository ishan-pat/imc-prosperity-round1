"""
Round 3 EDA — v2. Adds what v1 didn't have:

  T1  ADF stationarity test on log-returns (HYDROGEL_PACK and VELVETFRUIT_EXTRACT).
  T2  Engle-Granger cointegration test between the two underlyings.
  T3  Realized vol — rolling-window σ_per_day, to size A-S inventory penalty.
  T4  A-S calibration: estimate k (limit-order arrival intensity) from trade tape.
  T5  Raw SVI fit per day on the voucher slice (Gatheral & Jacquier 2014).
  T6  SVI parameter stability across days 0/1/2.
  T7  Sinclair: autocorrelation of smile residuals — are mispricings tradeable?

Run:
    python3 -m notebooks.round3_eda_v2
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint

DATA_DIR = REPO_ROOT / "data" / "round3"
PRICE_FILES = [
    (DATA_DIR / "prices_round_3_day_0.csv", 0, 8),
    (DATA_DIR / "prices_round_3_day_1.csv", 1, 7),
    (DATA_DIR / "prices_round_3_day_2.csv", 2, 6),
]
TRADE_FILES = {
    0: DATA_DIR / "trades_round_3_day_0.csv",
    1: DATA_DIR / "trades_round_3_day_1.csv",
    2: DATA_DIR / "trades_round_3_day_2.csv",
}

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TICKS_PER_DAY = 10000
DAY_LENGTH_TS = 1_000_000


# ─── data loaders (same as v1, kept local for self-contained run) ──────────

def load_ticks(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts = int(row["timestamp"])
            prod = row["product"]
            try:
                mid = float(row["mid_price"]) if row["mid_price"] else None
            except ValueError:
                mid = None
            bids, asks = {}, {}
            for lv in "123":
                bp, bv = row.get(f"bid_price_{lv}", ""), row.get(f"bid_volume_{lv}", "")
                ap, av = row.get(f"ask_price_{lv}", ""), row.get(f"ask_volume_{lv}", "")
                try:
                    if bp and bv: bids[int(float(bp))] = int(bv)
                    if ap and av: asks[int(float(ap))] = -int(av)
                except ValueError:
                    pass
            out.setdefault(ts, {})[prod] = {"mid": mid, "bids": bids, "asks": asks}
    return out


def load_trades(path):
    out = defaultdict(list)
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                ts = int(row["timestamp"]); price = float(row["price"]); qty = int(row["quantity"])
            except (ValueError, KeyError):
                continue
            out[ts].append({"sym": row["symbol"], "price": price, "qty": qty})
    return out


def mids_for(ticks, prod):
    out = []
    for ts in sorted(ticks):
        d = ticks[ts].get(prod)
        if d and d["mid"] is not None:
            out.append(d["mid"])
    return out


# ─── BS / IV (same as trader.py) ───────────────────────────────────────────

def _phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    sT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sT)
    d2 = d1 - sigma * sT
    return S * _phi(d1) - K * _phi(d2)


def bs_iv(price, S, K, T, lo=1e-5, hi=5.0, tol=1e-6):
    intrinsic = max(0.0, S - K)
    if price < intrinsic - 1e-6 or price > S + 1e-6:
        return float("nan")
    f_lo = bs_call(S, K, T, lo) - price
    f_hi = bs_call(S, K, T, hi) - price
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f = bs_call(S, K, T, m) - price
        if abs(f) < tol: return m
        if f > 0: hi = m
        else: lo = m
    return 0.5 * (lo + hi)


def tte_at(ts, day_start_tte):
    return day_start_tte - ts / DAY_LENGTH_TS


# ─── Raw SVI (Gatheral & Jacquier 2014, eq. 1) ─────────────────────────────
#
# w(k; a, b, ρ, m, σ) = a + b · (ρ·(k - m) + sqrt((k - m)^2 + σ^2))
#
# k = log-moneyness ln(K/F) (we use ln(K/S) since r=0 implies F=S),
# w = total implied variance σ_BS² · T.
# Constraints (as soft penalties): b ≥ 0, |ρ| ≤ 1, σ > 0, a + b·σ·sqrt(1-ρ²) ≥ 0.

def svi_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(strikes_logmoney, total_variances, x0=None, max_iter=200):
    """Fit raw-SVI to (k_i, w_i) pairs.
    Returns (params, rmse) or (None, inf) on failure."""
    k = np.asarray(strikes_logmoney, dtype=float)
    w = np.asarray(total_variances, dtype=float)
    finite = np.isfinite(w) & np.isfinite(k)
    k, w = k[finite], w[finite]
    if len(k) < 5:
        return None, float("inf")

    def loss(params):
        a, b, rho, m, sigma = params
        # Soft constraints
        pen = 0.0
        if b < 0: pen += 1e6 * b * b
        if abs(rho) > 0.999: pen += 1e6 * (abs(rho) - 0.999) ** 2
        if sigma <= 1e-6: pen += 1e6 * (1e-6 - sigma) ** 2
        # min total variance ≥ 0 (no-arb floor): a + b·σ·sqrt(1-ρ²) ≥ 0
        floor = a + b * sigma * math.sqrt(max(0.0, 1 - rho * rho))
        if floor < 0: pen += 1e6 * floor * floor
        w_hat = svi_raw(k, a, b, rho, m, sigma)
        return float(np.sum((w_hat - w) ** 2)) + pen

    if x0 is None:
        # data-driven initial guess
        x0 = (max(np.min(w), 1e-6), 0.05, 0.0, 0.0, 0.1)

    best = None
    best_loss = float("inf")
    # Multi-start to avoid local minima
    starts = [x0,
              (np.median(w), 0.10, -0.3, 0.0, 0.20),
              (np.median(w), 0.05, +0.3, 0.0, 0.10),
              (np.min(w), 0.20, 0.0, 0.0, 0.05)]
    for s in starts:
        try:
            r = minimize(loss, s, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": max_iter})
            if r.fun < best_loss:
                best_loss, best = r.fun, r.x
        except Exception:
            continue
    if best is None:
        return None, float("inf")
    w_hat = svi_raw(k, *best)
    rmse = float(np.sqrt(np.mean((w_hat - w) ** 2)))
    return tuple(best), rmse


# ─── T1: ADF stationarity ──────────────────────────────────────────────────

def t1_adf():
    print("=" * 90)
    print("T1 — ADF stationarity test on mid-price level and log-returns")
    print("=" * 90)
    print(f"{'Prod':<22} {'Day':>3} {'series':<10} {'ADF stat':>10} {'p':>10} "
          f"{'verdict':<22}")
    print("-" * 90)
    for prod in (HYDROGEL, UNDERLYING):
        for path, day, _ in PRICE_FILES:
            ticks = load_ticks(path)
            mids = mids_for(ticks, prod)
            if len(mids) < 100:
                continue
            log_rets = np.diff(np.log(mids))
            for series_name, x in (("level", np.array(mids)),
                                   ("log-rets", log_rets)):
                # use AIC-selected lag, regression='c' (constant only)
                stat, p, *_ = adfuller(x, regression='c', autolag='AIC')
                verdict = "stationary" if p < 0.05 else "unit root"
                print(f"{prod:<22} {day:>3} {series_name:<10} "
                      f"{stat:>10.3f} {p:>10.4f}  {verdict:<22}")
        print()


# ─── T2: Engle-Granger cointegration ───────────────────────────────────────

def t2_cointegration():
    print("=" * 90)
    print("T2 — Engle-Granger cointegration: HYDROGEL_PACK ~ VELVETFRUIT_EXTRACT")
    print("=" * 90)
    print(f"{'Day':>3}  {'EG stat':>10}  {'p':>10}  {'half-life':>12}  {'verdict':<30}")
    print("-" * 90)
    for path, day, _ in PRICE_FILES:
        ticks = load_ticks(path)
        # Align by timestamp
        ts_sorted = sorted(ticks)
        x_arr, y_arr = [], []
        for ts in ts_sorted:
            d = ticks[ts]
            if HYDROGEL in d and UNDERLYING in d:
                if d[HYDROGEL]["mid"] is not None and d[UNDERLYING]["mid"] is not None:
                    x_arr.append(d[UNDERLYING]["mid"])
                    y_arr.append(d[HYDROGEL]["mid"])
        x = np.array(x_arr); y = np.array(y_arr)
        if len(x) < 100:
            continue
        stat, p, _ = coint(y, x, trend='c', autolag='AIC')
        # OLS β for half-life:
        beta = np.cov(x, y)[0, 1] / np.var(x)
        alpha = y.mean() - beta * x.mean()
        residual = y - (alpha + beta * x)
        # AR(1) on residual → half-life
        r_t, r_tm1 = residual[1:], residual[:-1]
        b1 = np.cov(r_t, r_tm1)[0, 1] / np.var(r_tm1)
        if b1 > 0 and b1 < 1:
            hl = -math.log(2) / math.log(b1)
            hl_str = f"{hl:.0f} ticks"
        else:
            hl_str = "n/a"
        verdict = ("cointegrated, tradeable" if (p < 0.05 and b1 > 0 and b1 < 0.999)
                   else "no useful pairs alpha")
        print(f"{day:>3}  {stat:>10.3f}  {p:>10.4f}  {hl_str:>12}  {verdict:<30}")


# ─── T3: realized vol (rolling) ────────────────────────────────────────────

def t3_realized_vol():
    print("=" * 90)
    print("T3 — Realized vol of VELVETFRUIT_EXTRACT (per-day, rolling 1000 ticks)")
    print("=" * 90)
    print(f"{'Day':>3}  {'σ_perday(full)':>15}  {'σ_min(roll)':>13}  {'σ_max(roll)':>13}  "
          f"{'σ_p10':>10}  {'σ_p90':>10}")
    print("-" * 90)
    for path, day, _ in PRICE_FILES:
        ticks = load_ticks(path)
        mids = mids_for(ticks, UNDERLYING)
        log_rets = np.diff(np.log(mids))
        sigma_full = np.std(log_rets, ddof=1) * math.sqrt(TICKS_PER_DAY)
        # rolling σ_per_day from 1000-tick window
        rolling = []
        W = 1000
        for i in range(W, len(log_rets)):
            rolling.append(np.std(log_rets[i - W:i], ddof=1) * math.sqrt(TICKS_PER_DAY))
        rolling = np.array(rolling)
        print(f"{day:>3}  {sigma_full:>15.5f}  {rolling.min():>13.5f}  "
              f"{rolling.max():>13.5f}  {np.percentile(rolling, 10):>10.5f}  "
              f"{np.percentile(rolling, 90):>10.5f}")
    print("Interpretation:")
    print("  Use σ_perday ≈ 0.022 as the diffusion vol input to Avellaneda-Stoikov.")
    print("  Wide rolling range → σ is non-stationary → consider EWMA-σ in trader.")


# ─── T4: A-S k (LOB arrival intensity) ─────────────────────────────────────

def t4_arrival_intensity():
    print("=" * 90)
    print("T4 — A-S limit-order arrival intensity k from trade tape")
    print("=" * 90)
    print("k models P(fill at distance δ from mid) ∝ exp(-k·δ); fitted from")
    print("the empirical distribution of (trade_price - mid)/spread.")
    print()
    print(f"{'Prod':<22} {'Day':>3} {'N trades':>9} {'mean |δ|':>10} {'k̂':>10}")
    print("-" * 90)
    for prod in (HYDROGEL, UNDERLYING):
        for path, day, _ in PRICE_FILES:
            ticks = load_ticks(path)
            trades = load_trades(TRADE_FILES[day])
            offsets = []
            for ts in sorted(trades):
                d = ticks.get(ts, {}).get(prod)
                if not d or d["mid"] is None:
                    continue
                mid = d["mid"]
                for tr in trades[ts]:
                    if tr["sym"] != prod:
                        continue
                    offsets.append(abs(tr["price"] - mid))
            if not offsets:
                print(f"{prod:<22} {day:>3} {0:>9}  no trades")
                continue
            offsets = np.array(offsets)
            mean_off = offsets.mean()
            k_hat = 1.0 / mean_off if mean_off > 0 else float("inf")
            print(f"{prod:<22} {day:>3} {len(offsets):>9} "
                  f"{mean_off:>10.3f} {k_hat:>10.3f}")
        print()
    print("k̂ is the rate parameter of an exponential fit to |trade_price - mid|.")
    print("In A-S, optimal half-spread = γσ²(T-t) + (1/γ)·ln(1 + γ/k̂).")


# ─── T5/T6: SVI fit per day + parameter stability ──────────────────────────

def _avg_underlying(ticks):
    mids = mids_for(ticks, UNDERLYING)
    return float(np.mean(mids)) if mids else float("nan")


def _avg_iv_per_strike(ticks, day_start_tte, sample_every=100):
    """For each strike: average mid-IV over the day. Returns {K: (iv, n)}."""
    iv_acc = {K: [] for K in STRIKES}
    for ts in sorted(ticks)[::sample_every]:
        u = ticks[ts].get(UNDERLYING)
        if not u or u["mid"] is None:
            continue
        S = u["mid"]
        T = tte_at(ts, day_start_tte)
        if T <= 0:
            continue
        for K in STRIKES:
            d = ticks[ts].get(f"VEV_{K}")
            if not d or d["mid"] is None:
                continue
            iv = bs_iv(d["mid"], S, K, T)
            if iv == iv:
                iv_acc[K].append(iv)
    return {K: (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
            for K, v in iv_acc.items()}


def t5_svi_fit():
    print("=" * 90)
    print("T5 — Raw-SVI fit per day (total variance form, w=σ²·T)")
    print("=" * 90)
    print("       a            b          ρ          m         σ_svi      RMSE     "
          "N_strikes")
    print("-" * 90)
    fits = {}
    for path, day, tte0 in PRICE_FILES:
        ticks = load_ticks(path)
        S_avg = _avg_underlying(ticks)
        per_strike = _avg_iv_per_strike(ticks, tte0)
        # Build (k, w) pairs. Use mid-day TTE as representative T.
        T_day = tte0 - 0.5
        ks, ws = [], []
        for K in STRIKES:
            iv, n = per_strike[K]
            if iv != iv or n < 30:
                continue
            ks.append(math.log(K / S_avg))
            ws.append(iv * iv * T_day)  # total variance
        params, rmse = fit_svi(ks, ws)
        if params is None:
            print(f"Day {day}: SVI fit failed ({len(ks)} strikes)")
            continue
        a, b, rho, m, sg = params
        fits[day] = (params, rmse, S_avg, T_day, ks, ws)
        print(f"Day {day}: {a:>10.6f} {b:>10.5f} {rho:>10.3f} {m:>10.3f} "
              f"{sg:>10.4f}  {rmse:>10.6f}  {len(ks):>5}")
    return fits


def t6_svi_stability(fits):
    print()
    print("=" * 90)
    print("T6 — SVI parameter stability across days 0/1/2")
    print("=" * 90)
    if len(fits) < 2:
        print("not enough fits")
        return
    print(f"  param      day0       day1       day2     range/median")
    print("  " + "-" * 60)
    for i, name in enumerate(("a", "b", "ρ", "m", "σ")):
        vals = [fits[d][0][i] for d in sorted(fits)]
        med = np.median(vals)
        ratio = (max(vals) - min(vals)) / abs(med) if abs(med) > 1e-12 else float("inf")
        print(f"  {name:<6}  " + "  ".join(f"{v:>9.5f}" for v in vals)
              + f"  {ratio:>10.3f}")
    print()
    print("Interpretation:")
    print("  Small range/median → smile shape is stationary; can use day-avg params.")
    print("  Large range → fit per tick (or daily) and treat smile as time-varying.")


# ─── T7: Sinclair — autocorrelation of smile residuals ─────────────────────

def t7_residual_autocorr(fits):
    print()
    print("=" * 90)
    print("T7 — Sinclair test: are smile residuals tradeable (autocorrelated)?")
    print("=" * 90)
    print("For each strike, residual_t = (mid_IV_t)² · T - SVI_w(k_K). If ρ_lag1 ≈ 0")
    print("residuals are noise (don't trade). If ρ ≠ 0 they're forecastable.")
    print()
    print(f"  Day  Strike  N    σ_resid    AC1      half-life")
    print("  " + "-" * 60)
    for day in sorted(fits):
        params, rmse, S_avg, T_day, _, _ = fits[day]
        path = PRICE_FILES[day][0]
        tte0 = PRICE_FILES[day][2]
        ticks = load_ticks(path)
        sample = sorted(ticks)[::100]
        for K in STRIKES:
            res_series = []
            for ts in sample:
                u = ticks[ts].get(UNDERLYING)
                d = ticks[ts].get(f"VEV_{K}")
                if not u or not d or u["mid"] is None or d["mid"] is None:
                    continue
                S = u["mid"]
                T = tte_at(ts, tte0)
                if T <= 0:
                    continue
                iv = bs_iv(d["mid"], S, K, T)
                if iv != iv:
                    continue
                k = math.log(K / S)
                w_obs = iv * iv * T
                w_svi = svi_raw(k, *params)
                res_series.append(w_obs - w_svi)
            if len(res_series) < 30:
                continue
            r = np.array(res_series)
            if np.std(r) < 1e-12:
                continue
            ac1 = np.corrcoef(r[1:], r[:-1])[0, 1]
            if 0 < ac1 < 1:
                hl = -math.log(2) / math.log(ac1)
                hl_str = f"{hl:>6.1f}"
            else:
                hl_str = "  n/a"
            print(f"  {day}    {K:>5}  {len(r):>4}  {np.std(r):>9.6f}  "
                  f"{ac1:>+6.3f}   {hl_str}")
        print()
    print("Interpretation:")
    print("  AC1 near 0 → residuals are noise. SVI gives fair value, don't trade")
    print("              the deviations as alpha (consistent with v1 finding).")
    print("  AC1 ≫ 0   → residuals are persistent → mean-reversion on tick scale.")


# ─── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t1_adf()
    print()
    t2_cointegration()
    print()
    t3_realized_vol()
    print()
    t4_arrival_intensity()
    print()
    fits = t5_svi_fit()
    t6_svi_stability(fits)
    t7_residual_autocorr(fits)
