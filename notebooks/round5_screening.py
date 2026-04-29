"""
Round 5 systematic screening — all 50 products, days 2/3/4.

Per-product per-day metrics (Lo-MacKinlay 1988 VR; Box-Jenkins ACF; FFT
spectral diagnostics; ADF stationarity). Per-category structural analysis
(correlation, PCA, Engle-Granger cointegration). Cross-day stability.

Outputs:
  writeups/round5_screening.csv             — raw per-day metrics
  writeups/round5_screening_aggregated.csv  — averaged + ranked

Usage (from repo root):
    python3 -m notebooks.round5_screening
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "round5"
OUT_RAW = REPO_ROOT / "writeups" / "round5_screening.csv"
OUT_AGG = REPO_ROOT / "writeups" / "round5_screening_aggregated.csv"

CATEGORIES = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
                      "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SLEEP_POD":     ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
                      "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
    "MICROCHIP":     ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
                      "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"],
    "PEBBLES":       ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT":         ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
                      "ROBOT_LAUNDRY", "ROBOT_IRONING"],
    "UV_VISOR":      ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
                      "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "TRANSLATOR":    ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
                      "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
                      "TRANSLATOR_VOID_BLUE"],
    "PANEL":         ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
                      "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
                      "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
}


# ── per-product metrics ────────────────────────────────────────────────────

def variance_ratio(log_prices: np.ndarray, q: int) -> float:
    """Lo-MacKinlay 1988 VR(q). VR<1 mean-rev; VR>1 momentum; ≈1 random walk."""
    rets = np.diff(log_prices)
    if len(rets) < q + 10:
        return float("nan")
    var_1 = np.var(rets, ddof=1)
    if var_1 == 0:
        return float("nan")
    long_rets = log_prices[q:] - log_prices[:-q]
    var_q = np.var(long_rets, ddof=1)
    return float(var_q / (q * var_1))


def autocorr(rets: np.ndarray, lag: int) -> float:
    if len(rets) <= lag + 10:
        return float("nan")
    return float(np.corrcoef(rets[:-lag], rets[lag:])[0, 1])


def adf_p(prices: np.ndarray) -> float:
    try:
        # subsample for speed
        if len(prices) > 5000:
            idx = np.linspace(0, len(prices) - 1, 5000).astype(int)
            prices = prices[idx]
        return float(adfuller(np.log(prices), autolag="AIC")[1])
    except Exception:
        return float("nan")


def fft_peak(rets: np.ndarray):
    """Return (dominant_period, peak/median magnitude ratio).
    Box-Jenkins / Lo-Mamaysky-Wang style spectral diagnostic.
    Ratio >> 1 indicates a non-trivial cyclical component."""
    n = len(rets)
    if n < 200:
        return float("nan"), float("nan")
    centered = rets - rets.mean()
    fft = np.fft.rfft(centered)
    mag = np.abs(fft)
    if len(mag) <= 2:
        return float("nan"), float("nan")
    mag = mag[1:]  # drop DC
    peak_idx = int(np.argmax(mag))
    peak_period = n / (peak_idx + 1)
    med = float(np.median(mag))
    ratio = float(mag[peak_idx] / med) if med > 0 else float("nan")
    return float(peak_period), ratio


def screen_product_day(prod: str, day: int, df: pd.DataFrame) -> dict:
    mids = df["mid_price"].dropna().values
    mids = mids[(mids > 100) & (mids < 100000)]
    out = {"product": prod, "day": day, "n": len(mids)}
    if len(mids) < 1000:
        return out

    log_mids = np.log(mids)
    rets = np.diff(log_mids)

    out["mean_mid"] = float(np.mean(mids))
    out["std_mid"] = float(np.std(mids))
    out["rvol_bps"] = float(np.std(rets) * 10000)
    out["adf_p"] = adf_p(mids)
    out["vr2"] = variance_ratio(log_mids, 2)
    out["vr5"] = variance_ratio(log_mids, 5)
    out["vr10"] = variance_ratio(log_mids, 10)
    out["ac1"] = autocorr(rets, 1)
    out["ac5"] = autocorr(rets, 5)
    out["ac10"] = autocorr(rets, 10)
    out["ac50"] = autocorr(rets, 50)
    out["ac100"] = autocorr(rets, 100)
    period, ratio = fft_peak(rets)
    out["fft_period"] = period
    out["fft_ratio"] = ratio

    valid = df.dropna(subset=["bid_price_1", "ask_price_1"])
    if len(valid):
        spreads = (valid["ask_price_1"].values - valid["bid_price_1"].values)
        out["mean_spread"] = float(np.mean(spreads))
        out["mean_bid_size"] = float(valid["bid_volume_1"].mean()) if "bid_volume_1" in valid else float("nan")
        out["mean_ask_size"] = float(valid["ask_volume_1"].mean()) if "ask_volume_1" in valid else float("nan")
    else:
        out["mean_spread"] = float("nan")
        out["mean_bid_size"] = float("nan")
        out["mean_ask_size"] = float("nan")
    return out


# ── per-category structural analysis ───────────────────────────────────────

def category_analysis(prices: pd.DataFrame, members: list) -> dict:
    pivot = prices[prices["product"].isin(members)].pivot_table(
        index=["day", "timestamp"], columns="product", values="mid_price"
    ).dropna()
    if pivot.shape[0] < 1000 or pivot.shape[1] < 5:
        return {}

    cols = list(pivot.columns)
    log_p = np.log(pivot.values)
    rets = np.diff(log_p, axis=0)

    # Correlation
    corr = np.corrcoef(rets.T)
    avg_corr = float(corr[np.triu_indices(len(cols), k=1)].mean())

    # PCA via eigendecomposition of the standardized return covariance
    rets_std = (rets - rets.mean(axis=0)) / (rets.std(axis=0) + 1e-12)
    cov = np.cov(rets_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    var_exp = eigvals / eigvals.sum()

    # Pairwise Engle-Granger cointegration (subsample for speed)
    sub_idx = np.linspace(0, pivot.shape[0] - 1, min(pivot.shape[0], 5000)).astype(int)
    sub = pivot.values[sub_idx]
    coint_results = []
    for i, j in combinations(range(len(cols)), 2):
        try:
            _, p, _ = coint(sub[:, i], sub[:, j])
            coint_results.append((cols[i], cols[j], float(p)))
        except Exception:
            pass

    return {
        "members": cols,
        "avg_corr": avg_corr,
        "pc_var": var_exp.tolist(),
        "pc_loadings": eigvecs.tolist(),
        "coint": sorted(coint_results, key=lambda x: x[2]),
    }


# ── main pipeline ──────────────────────────────────────────────────────────

def main():
    print("Loading Round 5 price data...")
    dfs = []
    for f in sorted(DATA.glob("prices_round_5_day_*.csv")):
        dfs.append(pd.read_csv(f, sep=";"))
    prices = pd.concat(dfs, ignore_index=True)
    print(f"  {len(prices):,} rows, {prices['product'].nunique()} products, "
          f"days = {sorted(prices['day'].unique())}")

    print("\nScreening 50 products × 3 days...")
    rows = []
    for prod in sorted(prices["product"].unique()):
        sub = prices[prices["product"] == prod]
        for day in sorted(sub["day"].unique()):
            rows.append(screen_product_day(prod, int(day), sub[sub["day"] == day]))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_RAW, index=False)
    print(f"  Raw per-day metrics → {OUT_RAW.relative_to(REPO_ROOT)}")

    # Aggregate across days
    metric_cols = [c for c in raw.columns if c not in ("product", "day", "n")]
    agg = raw.groupby("product")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["product"] + [f"{c}_{stat}" for c, stat in agg.columns[1:]]
    agg.to_csv(OUT_AGG, index=False)
    print(f"  Aggregated metrics → {OUT_AGG.relative_to(REPO_ROOT)}")

    return raw, agg, prices


def report(raw, agg, prices):
    # Convenience aliases on the aggregated frame
    df = agg.copy()
    df["product"] = df["product"]

    print("\n" + "=" * 100)
    print("MASTER TABLE: per-product summary (averages across days)")
    print("=" * 100)
    cols_show = ["product", "rvol_bps_mean", "mean_spread_mean", "adf_p_mean",
                 "vr2_mean", "vr10_mean", "ac1_mean", "fft_ratio_mean", "fft_period_mean"]
    cols_show = [c for c in cols_show if c in df.columns]
    fmt = df[cols_show].copy()
    fmt = fmt.rename(columns={c: c.replace("_mean", "") for c in cols_show if c.endswith("_mean")})
    print(fmt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("RANKINGS (averaged across 3 days)")
    print("=" * 100)

    print("\n[A] Most mean-reverting (lowest VR(10), AC1<0):")
    top = df.sort_values("vr10_mean").head(12)
    for _, r in top.iterrows():
        print(f"  {r['product']:<35} VR(10)={r['vr10_mean']:.3f}  AC1={r['ac1_mean']:.3f}  "
              f"ADF p={r['adf_p_mean']:.3f}")

    print("\n[B] Strongest |AC1| (lag-1 predictability):")
    df["abs_ac1"] = df["ac1_mean"].abs()
    top = df.sort_values("abs_ac1", ascending=False).head(12)
    for _, r in top.iterrows():
        sign = "+" if r["ac1_mean"] > 0 else "-"
        print(f"  {r['product']:<35} AC1={r['ac1_mean']:+.3f} ({sign})  VR(10)={r['vr10_mean']:.3f}")

    print("\n[C] Strongest FFT peak (cyclical / deterministic structure suspect):")
    top = df.sort_values("fft_ratio_mean", ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"  {r['product']:<35} FFT ratio={r['fft_ratio_mean']:.2f}  "
              f"period~{r['fft_period_mean']:.0f}  AC50={r['ac50_mean']:.3f}")

    print("\n[D] Tightest spread × low vol (MM candidates):")
    df["mm_score"] = df["mean_spread_mean"] / (df["rvol_bps_mean"] / 10000 * df["mean_mid_mean"])
    df_mm = df[df["rvol_bps_mean"] > 5].sort_values("mm_score").head(12)
    for _, r in df_mm.iterrows():
        print(f"  {r['product']:<35} spread={r['mean_spread_mean']:.2f}  "
              f"rvol={r['rvol_bps_mean']:.1f}bps  mm_score={r['mm_score']:.2f}")

    print("\n" + "=" * 100)
    print("WITHIN-CATEGORY STRUCTURAL ANALYSIS (Avellaneda-Lee 2010)")
    print("=" * 100)

    cat_results = {}
    for cat, members in CATEGORIES.items():
        info = category_analysis(prices, members)
        if not info:
            continue
        cat_results[cat] = info
        print(f"\n[{cat}]")
        print(f"  Avg pairwise correlation: {info['avg_corr']:.3f}")
        ve = info["pc_var"]
        print(f"  PCA variance: PC1={ve[0]:.1%}  PC2={ve[1]:.1%}  PC3={ve[2]:.1%}  "
              f"PC4={ve[3]:.1%}  PC5={ve[4]:.1%}")
        print(f"  Top 3 cointegrated pairs (Engle-Granger p):")
        for a, b, p in info["coint"][:3]:
            tag = "  *** SIGNIFICANT" if p < 0.05 else ("  marginal" if p < 0.10 else "")
            print(f"    {a:<33} ⇄ {b:<33} p={p:.4f}{tag}")
        # If PC1 explains >70%, this is an Avellaneda-Lee target category
        if ve[0] > 0.7:
            print(f"  ★ Avellaneda-Lee target: PC1 dominates — trade PC2-PC5 spreads")

    # Cross-sectional reversal/momentum preview (Lehmann 1990), per day for stability
    print("\n" + "=" * 100)
    print("CROSS-SECTIONAL LONG-WORST / SHORT-BEST (Lehmann 1990) per day")
    print("Positive IR ⇒ reversal works (long loser); negative ⇒ momentum (long winner)")
    print("=" * 100)
    print(f"{'category':<15}  {'day2 t-stat':>12}  {'day3 t-stat':>12}  {'day4 t-stat':>12}  "
          f"{'all3 t-stat':>12}  {'all3 bps/cycle':>15}")
    print("-" * 100)
    for cat, members in CATEGORIES.items():
        pivot = prices[prices["product"].isin(members)].pivot_table(
            index=["day", "timestamp"], columns="product", values="mid_price"
        ).dropna()
        if pivot.shape[0] < 1000:
            continue

        def reversal_t_stat(sub_pivot):
            if sub_pivot.shape[0] < 250:
                return float("nan"), float("nan"), 0
            log_p = np.log(sub_pivot.values)
            LB, FWD, STRIDE = 100, 100, 100  # non-overlapping
            pnls = []
            for t in range(LB, log_p.shape[0] - FWD, STRIDE):
                past = log_p[t] - log_p[t - LB]
                future = log_p[t + FWD] - log_p[t]
                ranks = past.argsort().argsort()
                pnl = future[ranks == 0][0] - future[ranks == 4][0]
                pnls.append(pnl)
            pnls = np.array(pnls)
            if len(pnls) < 5:
                return float("nan"), float("nan"), len(pnls)
            mu = pnls.mean()
            sd = pnls.std(ddof=1)
            t = mu / sd * np.sqrt(len(pnls)) if sd > 0 else 0
            return t, mu * 10000, len(pnls)

        per_day = []
        for d in [2, 3, 4]:
            sub = pivot[pivot.index.get_level_values("day") == d]
            t, mu_bps, n = reversal_t_stat(sub)
            per_day.append((t, mu_bps, n))
        t_all, mu_all, n_all = reversal_t_stat(pivot)
        d2t, d3t, d4t = per_day[0][0], per_day[1][0], per_day[2][0]
        # Stability flag: all 3 days same sign
        signs = [np.sign(t) for t in (d2t, d3t, d4t) if not np.isnan(t)]
        stable = "★ STABLE" if len(signs) == 3 and len(set(signs)) == 1 else ""
        print(f"  {cat:<15}  {d2t:>+12.2f}  {d3t:>+12.2f}  {d4t:>+12.2f}  "
              f"{t_all:>+12.2f}  {mu_all:>+12.1f}    {stable}")


if __name__ == "__main__":
    raw, agg, prices = main()
    report(raw, agg, prices)
