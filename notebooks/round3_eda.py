"""
Round 3 EDA — drives the trader.py strategy.

Three days of historical data (TTE 8/7/6 at start of each day). Live round
will start at TTE = 5 days. Conventions per user spec:
  - r = 0
  - T in days (no annualization), σ per-day fitted from per-tick log returns
    scaled by sqrt(N_ticks_per_day=10000)

Sections:
  S1  Stationarity / AC / RV for HYDROGEL_PACK and VELVETFRUIT_EXTRACT
  S2  Voucher liquidity & quote sparsity per strike
  S3  IV smile per day (BS inverse, r=0); fit quadratic in log-moneyness;
      stability of fit across days
  S4  Voucher mispricings vs the fitted smile (which strikes are rich/cheap)
  S5  Forward-looking hints for strategy

Run:
    python3 -m notebooks.round3_eda
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "round3"

# (file, day_index, TTE_at_start_of_day_in_days)
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
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
TICKS_PER_DAY = 10000           # 1M ts / 100ms step
DAY_LENGTH_TS = 1_000_000


# ───────────────────────────────────────────────────────────────────────────
# loaders

def load_ticks(filename: Path) -> dict:
    """{timestamp: {product: {mid, bid1, ask1, bids, asks}}}"""
    out: dict = {}
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            prod = row["product"]
            try:
                mid = float(row["mid_price"]) if row["mid_price"] else None
            except ValueError:
                mid = None
            bids: dict = {}
            asks: dict = {}
            for level in ("1", "2", "3"):
                bp, bv = row.get(f"bid_price_{level}", ""), row.get(f"bid_volume_{level}", "")
                ap, av = row.get(f"ask_price_{level}", ""), row.get(f"ask_volume_{level}", "")
                if bp and bv:
                    try:
                        bids[int(float(bp))] = int(bv)
                    except ValueError:
                        pass
                if ap and av:
                    try:
                        asks[int(float(ap))] = -int(av)
                    except ValueError:
                        pass
            out.setdefault(ts, {})[prod] = {"mid": mid, "bids": bids, "asks": asks}
    return out


def load_trades(filename: Path) -> dict:
    out: dict = defaultdict(list)
    if not filename.exists():
        return out
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                ts = int(row["timestamp"])
                price = float(row["price"])
                qty = int(row["quantity"])
            except (ValueError, KeyError):
                continue
            out[ts].append({"symbol": row["symbol"], "price": price, "qty": qty})
    return out


def mids_for(ticks: dict, product: str):
    """[(ts, mid)] for product, ts-sorted, mids that exist."""
    out = []
    for ts in sorted(ticks):
        d = ticks[ts].get(product)
        if d and d["mid"] is not None:
            out.append((ts, d["mid"]))
    return out


def lag1_ac(x):
    if len(x) < 3:
        return float("nan")
    m = statistics.fmean(x)
    num = sum((x[i] - m) * (x[i - 1] - m) for i in range(1, len(x)))
    den = sum((xi - m) ** 2 for xi in x)
    return num / den if den > 0 else float("nan")


# ───────────────────────────────────────────────────────────────────────────
# Black-Scholes (r = 0, q = 0) and IV inverse via bisection

def _phi(x):  # standard-normal cdf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """T in days (per user's no-annualization convention); sigma in per-day."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _phi(d1) - K * _phi(d2)


def bs_iv(price: float, S: float, K: float, T: float,
          lo: float = 1e-5, hi: float = 5.0, tol: float = 1e-6) -> float:
    """Bisection IV solver. Returns NaN if price violates no-arb bounds."""
    intrinsic = max(0.0, S - K)
    if price < intrinsic - 1e-6 or price > S:
        return float("nan")
    f_lo = bs_call(S, K, T, lo) - price
    f_hi = bs_call(S, K, T, hi) - price
    if f_lo > 0:
        return float("nan")  # even at zero vol price is too high → arb / parity
    if f_hi < 0:
        return float("nan")  # vol > 5/day → walk away
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = bs_call(S, K, T, mid) - price
        if abs(f) < tol:
            return mid
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _phi(d1)


# ───────────────────────────────────────────────────────────────────────────
# S1 — underlying stationarity / vol

def s1_underlying_stats():
    print("=" * 92)
    print("S1 — HYDROGEL_PACK and VELVETFRUIT_EXTRACT mid-price stats per day")
    print("=" * 92)
    print(f"{'Prod':<22} {'Day':>3} {'N':>6} {'Mean':>10} {'Std':>9} "
          f"{'Range':>12} {'σ(Δ)':>9} {'AC1(Δ)':>9} {'σ_perday':>10} {'BPS/day':>9}")
    print("-" * 92)
    rv_summary = {}
    for product in (HYDROGEL, UNDERLYING):
        for price_file, day, _ in PRICE_FILES:
            ticks = load_ticks(price_file)
            series = mids_for(ticks, product)
            mids = [m for _, m in series]
            if len(mids) < 10:
                continue
            diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
            log_rets = [math.log(mids[i] / mids[i - 1]) for i in range(1, len(mids)) if mids[i - 1] > 0]
            mean = statistics.fmean(mids)
            std = statistics.stdev(mids)
            sigma_diff = statistics.stdev(diffs)
            ac1 = lag1_ac(diffs)
            # per-day σ of log-returns: σ_perday = σ_pertick * sqrt(N_ticks_per_day)
            sigma_perday = statistics.stdev(log_rets) * math.sqrt(TICKS_PER_DAY) if log_rets else float("nan")
            print(f"{product:<22} {day:>3} {len(mids):>6} {mean:>10.2f} {std:>9.2f} "
                  f"{min(mids):>5.0f}–{max(mids):<5.0f} {sigma_diff:>9.3f} "
                  f"{ac1:>9.3f} {sigma_perday:>10.4f} {sigma_perday * 10000:>9.1f}")
            if product == UNDERLYING:
                rv_summary[day] = sigma_perday
        print()
    print("Interpretation hint:")
    print("  AC1(Δ)  ≪ 0       → strong mean-reversion → MM with tight reservation skew")
    print("  AC1(Δ)  ~ 0       → random-walk           → MM around mid, wide-ish reservations")
    print("  AC1(Δ)  > 0.2     → trending              → momentum / careful skew")
    print("  σ_perday is the input we'll feed to Black-Scholes (T in days, no annualization).")
    return rv_summary


# ───────────────────────────────────────────────────────────────────────────
# S2 — voucher liquidity / sparsity

def s2_voucher_liquidity():
    print("=" * 92)
    print("S2 — Voucher quote presence and depth per strike (avg over day 0–2)")
    print("=" * 92)
    print(f"{'Strike':>7} {'Symbol':<10} {'Day':>3} {'Quoted%':>9} "
          f"{'AvgBid':>10} {'AvgAsk':>10} {'AvgSpread':>10} {'AvgVolBest':>11}")
    print("-" * 92)
    for k, sym in zip(STRIKES, VOUCHERS):
        for price_file, day, _ in PRICE_FILES:
            ticks = load_ticks(price_file)
            n_ts = len(ticks)
            quoted = 0
            spreads, mids, vols = [], [], []
            bids_s, asks_s = [], []
            for ts in ticks:
                d = ticks[ts].get(sym)
                if not d:
                    continue
                bb = max(d["bids"]) if d["bids"] else None
                ba = min(d["asks"]) if d["asks"] else None
                if bb is not None and ba is not None:
                    quoted += 1
                    spreads.append(ba - bb)
                    mids.append(0.5 * (bb + ba))
                    bids_s.append(bb)
                    asks_s.append(ba)
                    vols.append(min(d["bids"][bb], -d["asks"][ba]))
            if quoted == 0:
                print(f"{k:>7} {sym:<10} {day:>3}  no quotes")
                continue
            print(f"{k:>7} {sym:<10} {day:>3} "
                  f"{100 * quoted / n_ts:>8.1f}% "
                  f"{statistics.fmean(bids_s):>10.2f} "
                  f"{statistics.fmean(asks_s):>10.2f} "
                  f"{statistics.fmean(spreads):>10.2f} "
                  f"{statistics.fmean(vols):>11.1f}")
        print()
    print("Interpretation hint:")
    print("  Strikes with <50% quoting are not tradeable for active vol-arb.")
    print("  Wide avg spread → high adverse-selection cost → only trade on big mispricing.")


# ───────────────────────────────────────────────────────────────────────────
# helper — TTE at given timestamp within a day

def tte_days_at(timestamp: int, day_start_tte: float) -> float:
    """Linear decay across the day. day_start_tte is in days (e.g. 5).
    At ts=0  → tte = day_start_tte
    At ts=DAY_LENGTH_TS → tte = day_start_tte - 1
    """
    progress = timestamp / DAY_LENGTH_TS
    return day_start_tte - progress


# ───────────────────────────────────────────────────────────────────────────
# S3 — IV smile per day, smile stability

def _avg_smile_for_day(ticks: dict, day_start_tte: float, sample_every: int = 100):
    """Average IV per strike over the day (sampled to keep things fast)."""
    iv_acc = {k: [] for k in STRIKES}
    sampled = sorted(ticks)[::sample_every]
    for ts in sampled:
        d_under = ticks[ts].get(UNDERLYING)
        if not d_under or d_under["mid"] is None:
            continue
        S = d_under["mid"]
        T = tte_days_at(ts, day_start_tte)
        if T <= 0:
            continue
        for k in STRIKES:
            sym = f"VEV_{k}"
            d = ticks[ts].get(sym)
            if not d or d["mid"] is None:
                continue
            iv = bs_iv(d["mid"], S, k, T)
            if iv == iv:  # not NaN
                iv_acc[k].append(iv)
    return {k: (statistics.fmean(v) if v else float("nan"),
                statistics.stdev(v) if len(v) > 1 else float("nan"),
                len(v))
            for k, v in iv_acc.items()}


def s3_iv_smile():
    print("=" * 92)
    print("S3 — Implied vol per strike per day (BS, r=0, T in days, σ per-day)")
    print("=" * 92)
    print(f"{'Strike':>7}  {'Day0 IV':>10}  {'Day1 IV':>10}  {'Day2 IV':>10}  "
          f"{'Mean':>8}  {'Std':>8}  {'N0/N1/N2':>14}")
    print("-" * 92)
    smiles = {}
    for price_file, day, tte0 in PRICE_FILES:
        ticks = load_ticks(price_file)
        smiles[day] = _avg_smile_for_day(ticks, tte0)
    for k in STRIKES:
        ivs = []
        ns = []
        for d in (0, 1, 2):
            mu, _, n = smiles[d].get(k, (float("nan"), float("nan"), 0))
            ivs.append(mu)
            ns.append(n)
        valid = [v for v in ivs if v == v]
        mu_all = statistics.fmean(valid) if valid else float("nan")
        std_all = statistics.stdev(valid) if len(valid) > 1 else float("nan")
        print(f"{k:>7}  "
              f"{ivs[0]:>10.5f}  {ivs[1]:>10.5f}  {ivs[2]:>10.5f}  "
              f"{mu_all:>8.5f}  {std_all:>8.5f}  "
              f"{ns[0]:>4}/{ns[1]:>4}/{ns[2]:>4}")
    print()
    print("Interpretation hint:")
    print("  σ ≈ stable across days  → vol smile is structural; trade deviations from fit.")
    print("  σ rising with TTE↓     → vouchers price in vol crush near expiry; be careful.")

    # Quadratic smile fit on day 2 (closest to live TTE)
    print()
    print("Quadratic smile fit IV(m) = a + b·m + c·m² where m = ln(K/S_avg) (per day):")
    for day in (0, 1, 2):
        # collect (ln(K/S_avg), iv_mean) for strikes that have data
        S_proxy = _avg_underlying(load_ticks(PRICE_FILES[day][0]))
        xs, ys = [], []
        for k in STRIKES:
            mu, _, n = smiles[day].get(k, (float("nan"), float("nan"), 0))
            if mu == mu and n >= 30:
                xs.append(math.log(k / S_proxy))
                ys.append(mu)
        if len(xs) < 4:
            continue
        a, b, c = _fit_quadratic(xs, ys)
        rmse = _rmse(xs, ys, a, b, c)
        print(f"  Day {day}: S≈{S_proxy:.1f}  IV(m)= {a:.5f} + {b:+.5f}·m + {c:+.5f}·m²  "
              f"RMSE={rmse:.5f}  N_strikes_used={len(xs)}")
    return smiles


def _avg_underlying(ticks: dict) -> float:
    mids = [m for _, m in mids_for(ticks, UNDERLYING)]
    return statistics.fmean(mids) if mids else float("nan")


def _fit_quadratic(xs, ys):
    """Least-squares y = a + b x + c x^2 via normal equations on a 3×3 system."""
    n = len(xs)
    Sx  = sum(xs);                Sy = sum(ys)
    Sx2 = sum(x * x for x in xs); Sxy = sum(x * y for x, y in zip(xs, ys))
    Sx3 = sum(x ** 3 for x in xs); Sx2y = sum(x * x * y for x, y in zip(xs, ys))
    Sx4 = sum(x ** 4 for x in xs)
    # Solve [[n, Sx, Sx2],[Sx, Sx2, Sx3],[Sx2, Sx3, Sx4]] · [a,b,c]ᵀ = [Sy, Sxy, Sx2y]
    M = [[n, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]]
    v = [Sy, Sxy, Sx2y]
    return _solve3(M, v)


def _solve3(M, v):
    """Cramer's rule on 3×3."""
    def det3(A):
        return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    D = det3(M)
    if abs(D) < 1e-18:
        return float("nan"), float("nan"), float("nan")
    out = []
    for col in range(3):
        Mc = [row[:] for row in M]
        for r in range(3):
            Mc[r][col] = v[r]
        out.append(det3(Mc) / D)
    return tuple(out)


def _rmse(xs, ys, a, b, c):
    s = 0.0
    for x, y in zip(xs, ys):
        e = (a + b * x + c * x * x) - y
        s += e * e
    return math.sqrt(s / len(xs))


# ───────────────────────────────────────────────────────────────────────────
# S4 — voucher mispricings vs fitted smile

def s4_mispricings():
    """For each strike, average premium-residual (market mid − BS price using fitted smile σ) per day.
    A consistent rich strike (positive residual every day) is a sell candidate."""
    print("=" * 92)
    print("S4 — Average market−model voucher premium per strike (uses day-specific smile fit)")
    print("=" * 92)
    print(f"{'Strike':>7}  {'Day0 mid_resid':>16}  {'Day1 mid_resid':>16}  {'Day2 mid_resid':>16}  "
          f"{'AvgResid':>10}  {'Sign':>6}")
    print("-" * 92)
    smiles = {}
    s_proxies = {}
    fits = {}
    for price_file, day, tte0 in PRICE_FILES:
        ticks = load_ticks(price_file)
        smiles[day] = _avg_smile_for_day(ticks, tte0)
        s_proxies[day] = _avg_underlying(ticks)
        xs, ys = [], []
        for k in STRIKES:
            mu, _, n = smiles[day].get(k, (float("nan"), float("nan"), 0))
            if mu == mu and n >= 30:
                xs.append(math.log(k / s_proxies[day]))
                ys.append(mu)
        fits[day] = _fit_quadratic(xs, ys) if len(xs) >= 4 else (float("nan"), 0.0, 0.0)

    rich_cheap = {k: [] for k in STRIKES}
    for price_file, day, tte0 in PRICE_FILES:
        ticks = load_ticks(price_file)
        a, b, c = fits[day]
        S_avg = s_proxies[day]
        if not (a == a):
            continue
        residuals = {k: [] for k in STRIKES}
        for ts in sorted(ticks)[::100]:
            d_und = ticks[ts].get(UNDERLYING)
            if not d_und or d_und["mid"] is None:
                continue
            S = d_und["mid"]
            T = tte_days_at(ts, tte0)
            if T <= 0:
                continue
            for k in STRIKES:
                sym = f"VEV_{k}"
                dv = ticks[ts].get(sym)
                if not dv or dv["mid"] is None:
                    continue
                m = math.log(k / S)
                sigma_fit = a + b * m + c * m * m
                if sigma_fit <= 0:
                    continue
                model_price = bs_call(S, k, T, sigma_fit)
                residuals[k].append(dv["mid"] - model_price)
        for k in STRIKES:
            rich_cheap[k].append(statistics.fmean(residuals[k]) if residuals[k] else float("nan"))

    for k in STRIKES:
        rs = rich_cheap[k]
        valid = [r for r in rs if r == r]
        avg = statistics.fmean(valid) if valid else float("nan")
        # consistent sign across days?
        signs = {1 if r > 0 else (-1 if r < 0 else 0) for r in valid}
        sign_str = "rich" if signs == {1} else ("cheap" if signs == {-1} else "mixed")
        rs_disp = [f"{r:>+16.3f}" if r == r else f"{'na':>16}" for r in rs]
        print(f"{k:>7}  {rs_disp[0]}  {rs_disp[1]}  {rs_disp[2]}  "
              f"{avg:>+10.3f}  {sign_str:>6}")
    print()
    print("Interpretation hint:")
    print("  Consistently 'rich' → sell candidate; 'cheap' → buy candidate.")
    print("  Magnitude in same units as voucher mid (XIRECs).")


# ───────────────────────────────────────────────────────────────────────────
# S5 — strategy hint summary (purely descriptive printout)

def s5_summary():
    print("=" * 92)
    print("S5 — Strategy hints (read in conjunction with S1–S4 numbers)")
    print("=" * 92)
    print("""
HYDROGEL_PACK
  • Read AC1 from S1. If |AC1| < 0.05 across days → random-walk; market-make
    around the mid with light inventory skew. If AC1 < -0.2 → mean-revert
    around a slow-moving fair (EWMA). If AC1 > 0 → momentum / avoid MM.
  • limit 200 → 4-level book at ±1, ±2, ±3, ±4 ticks with size 12/level (=48)
    on each side at flat inventory.

VELVETFRUIT_EXTRACT (vouchers' underlying)
  • This must trade: it's the delta-hedge instrument. Quote tight & wide.
  • limit 200, but reserve ~80 capacity for delta-hedging vouchers; market-make
    the residual capacity.

VOUCHERS
  • 10 strikes; only those with >= 60% quoting are reliably tradeable. Print
    that filter from S2.
  • Per-tick decision: compute smile (a,b,c) from last K observations of
    (m=ln(K/S), IV) per strike. For each strike, model_price = BS(S,K,T,σ_fit).
    If market_ask < model_price - δ → buy 1; if market_bid > model_price + δ → sell 1.
    δ = adverse-selection cushion; tune via backtest, start at 1 XIREC for
    dollar-cheap strikes, scale with vega.
  • Net delta: sum across positions: pos[VEV_K] · delta(S,K,T,σ_fit).
    Hedge with VELVETFRUIT_EXTRACT to keep |net_delta| < 30 contracts.
  • EOD flatten in last 5–10% of timestamps: liquidate all voucher inventory
    at any tradeable touch — settlement isn't free (round boundary penalty),
    and we cannot exit between rounds.
""")


# ───────────────────────────────────────────────────────────────────────────
# main

if __name__ == "__main__":
    s1_underlying_stats()
    print()
    s2_voucher_liquidity()
    print()
    s3_iv_smile()
    print()
    s4_mispricings()
    print()
    s5_summary()
