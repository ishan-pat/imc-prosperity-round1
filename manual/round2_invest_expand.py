"""
Round 2 Manual Challenge — Invest & Expand optimization.

Rules (verbatim from platform):
  Research(r) = 200_000 × ln(1+r) / ln(101)       # logarithmic, sat at 200k
  Scale(s)    = 7 × s / 100                        # linear, sat at 7
  Speed(v)    = 0.9 - 0.8 × (rank(v)-1)/(N-1)      # competition-rank, ties share rank
                                                     next distinct value skips tied ranks
                                                     rank 1 = top, rank N = bottom
  Budget_Used = 50_000 × (r+s+v) / 100
  PnL = Research × Scale × Speed − Budget_Used

Key observations driving the analysis:
  1. PnL is multiplicative. Research(0) = 0 and Scale(0) = 0 both kill PnL.
     Speed is always in [0.1, 0.9]; v=0 is legal and yields ≥ 0.1.
  2. Research saturates violently: r=1 gives 15% of max; r=20 gives 66%;
     r=100 gives 100%. Don't overspend.
  3. Scale is linear, so marginal value per percentage point is constant.
     Budget leftover after Research/Speed goes here.
  4. Speed depends on the crowd. We simulate 3 plausible crowd distributions.

Usage (from repo root):
    python3 -m manual.round2_invest_expand
"""
import math
import random
import statistics
import sys


# ──────────────────────────────────────────────────────────────────────────
# Pillar functions

def research(r: int) -> float:
    return 200_000 * math.log(1 + r) / math.log(101)


def scale(s: int) -> float:
    return 7 * s / 100


def competition_rank(my_v: int, crowd_v: list) -> int:
    """Competition rank: 1 + number of crowd entries strictly greater than my_v.
    Matches the rules example (70,70,70,50,40,40,30 → ranks 1,1,1,4,5,5,7)."""
    return 1 + sum(1 for c in crowd_v if c > my_v)


def speed_multiplier(my_v: int, crowd_v: list) -> float:
    """Linear interpolation of rank from 0.9 (rank 1) to 0.1 (rank N)."""
    n = len(crowd_v) + 1
    if n <= 1:
        return 0.5
    r = competition_rank(my_v, crowd_v)
    return 0.9 - 0.8 * (r - 1) / (n - 1)


def budget_used(r: int, s: int, v: int) -> int:
    return 500 * (r + s + v)   # = 50_000 × (r+s+v)/100


# ──────────────────────────────────────────────────────────────────────────
# Crowd model

CROWD_SCENARIOS = {
    "low":  {"mu": 15, "sigma": 12, "type": "normal",
             "desc": "Most teams under-allocate Speed; many bid 0–20."},
    "base": {"mu": 30, "sigma": 18, "type": "normal",
             "desc": "Balanced crowd; typical thirds-ish allocation (~30)."},
    "high": {"mu": 50, "sigma": 20, "type": "normal",
             "desc": "Competitive; teams recognize rank matters and push 40–70."},
    "mixed": {"type": "mixture",
              "desc": "Discord-informed mixture: 15% at v=0 (naive-zero), 10% at v=100 "
                      "(Lucid-style GTO), 60% balanced thirds N(30,10), 15% "
                      "moderate Speed-heavy N(55,10)."},
}


def sample_crowd(scenario: str, n_teams: int, rng: random.Random) -> list:
    """Draw (n_teams - 1) integer v-values in [0, 100] under the given scenario."""
    p = CROWD_SCENARIOS[scenario]
    out = []
    for _ in range(n_teams - 1):
        if p["type"] == "normal":
            x = int(round(rng.gauss(p["mu"], p["sigma"])))
        else:  # mixture
            u = rng.random()
            if u < 0.15:
                x = 0
            elif u < 0.75:   # 60% balanced
                x = int(round(rng.gauss(30, 10)))
            elif u < 0.90:   # 15% moderate
                x = int(round(rng.gauss(55, 10)))
            else:            # 10% all-in
                x = 100
        out.append(max(0, min(100, x)))
    return out


def estimate_expected_speed(scenario: str, n_teams: int, n_sims: int, seed: int = 42):
    """Return dict v → E[Speed(v; crowd)] averaged over n_sims crowd draws."""
    rng = random.Random(seed)
    crowds = [sample_crowd(scenario, n_teams, rng) for _ in range(n_sims)]
    e_speed = {}
    for v in range(101):
        e_speed[v] = statistics.mean(speed_multiplier(v, c) for c in crowds)
    return e_speed


# ──────────────────────────────────────────────────────────────────────────
# Grid search

def grid_search(e_speed: dict):
    """Enumerate all (r, s, v) with r ≥ 1, s ≥ 1, v ≥ 0, r+s+v ≤ 100.
    Returns list of (E[PnL], r, s, v) sorted descending."""
    results = []
    for r in range(1, 100):
        R = research(r)
        for s in range(1, 101 - r):
            S = scale(s)
            RS = R * S
            for v in range(0, 101 - r - s):
                revenue = RS * e_speed[v]
                cost = budget_used(r, s, v)
                results.append((revenue - cost, r, s, v))
    results.sort(reverse=True, key=lambda x: x[0])
    return results


# ──────────────────────────────────────────────────────────────────────────
# Reporting

def format_allocation(r: int, s: int, v: int, e_speed: dict) -> str:
    R = research(r)
    S = scale(s)
    Sp = e_speed[v]
    revenue = R * S * Sp
    cost = budget_used(r, s, v)
    return (f"r={r:3d} s={s:3d} v={v:3d}  "
            f"R={R:>7,.0f}  S={S:>4.2f}  E[Sp]={Sp:>4.2f}  "
            f"Rev={revenue:>9,.0f}  Cost={cost:>6,}  "
            f"E[PnL]={revenue - cost:>9,.0f}")


def main(n_teams: int = 500, n_sims: int = 100):
    print(f"Simulation parameters: n_teams={n_teams}, n_sims={n_sims}")
    print(f"Pillar function sanity checks:")
    print(f"  Research(1)={research(1):,.0f}, Research(10)={research(10):,.0f}, "
          f"Research(50)={research(50):,.0f}, Research(100)={research(100):,.0f}")
    print(f"  Scale(10)={scale(10)}, Scale(50)={scale(50)}, Scale(100)={scale(100)}")

    all_top = {}
    for scenario in ["low", "base", "high", "mixed"]:
        p = CROWD_SCENARIOS[scenario]
        print()
        print("=" * 100)
        print(f"CROWD SCENARIO: {scenario.upper()}  ({p['desc']})")
        if p["type"] == "normal":
            print(f"  crowd v ~ N({p['mu']}, {p['sigma']}) clipped [0,100]")
        print("=" * 100)
        e_speed = estimate_expected_speed(scenario, n_teams, n_sims)
        print(f"E[Speed(v)] samples:  "
              f"v=0→{e_speed[0]:.3f}  v=10→{e_speed[10]:.3f}  "
              f"v=25→{e_speed[25]:.3f}  v=50→{e_speed[50]:.3f}  "
              f"v=75→{e_speed[75]:.3f}  v=100→{e_speed[100]:.3f}")
        results = grid_search(e_speed)
        print(f"\nTop 10 allocations by E[PnL]:")
        for epnl, r, s, v in results[:10]:
            print(f"  {format_allocation(r, s, v, e_speed)}")
        all_top[scenario] = results[:50]

    # ------------------------------------------------------------------
    # Robust evaluation: for the union of top-100 candidates from each scenario,
    # compute E[PnL] under ALL three scenarios. Rank by three criteria:
    #   (a) equal-weight mean
    #   (b) worst-case (min across scenarios)
    #   (c) prior-weighted (0.25 low, 0.50 base, 0.25 high)
    print()
    print("=" * 100)
    print("ROBUST EVALUATION (union of top-100 candidates from each scenario)")
    print("=" * 100)

    # Candidate pool
    candidates = set()
    for scenario in ["low", "base", "high"]:
        for _, r, s, v in all_top[scenario][:100]:
            candidates.add((r, s, v))

    # Pre-computed E[Speed] per scenario (re-run to ensure consistent seeds)
    speed_by_sc = {}
    for scenario in ["low", "base", "high", "mixed"]:
        speed_by_sc[scenario] = estimate_expected_speed(scenario, n_teams, n_sims)

    rows = []
    for (r, s, v) in candidates:
        R = research(r)
        S = scale(s)
        cost = budget_used(r, s, v)
        pnls = {}
        for scenario in ["low", "base", "high", "mixed"]:
            pnls[scenario] = R * S * speed_by_sc[scenario][v] - cost
        avg = statistics.mean([pnls["low"], pnls["base"], pnls["high"]])
        worst = min([pnls["low"], pnls["base"], pnls["high"]])
        prior = 0.25 * pnls["low"] + 0.50 * pnls["base"] + 0.25 * pnls["high"]
        rows.append((r, s, v, pnls["low"], pnls["base"], pnls["high"], pnls["mixed"],
                     avg, worst, prior))

    def _print_top(label, key_idx, rows, k=10):
        print(f"\n{label}:")
        print(f"{'r':>3} {'s':>3} {'v':>3}   "
              f"{'PnL_low':>10}  {'PnL_base':>10}  {'PnL_high':>10}  "
              f"{'PnL_mix':>10}  {'mean':>10}  {'worst':>10}  {'prior':>10}")
        for row in sorted(rows, key=lambda x: -x[key_idx])[:k]:
            r, s, v, p_low, p_base, p_high, p_mixed, avg, worst, prior = row
            print(f"{r:>3} {s:>3} {v:>3}   "
                  f"{p_low:>10,.0f}  {p_base:>10,.0f}  {p_high:>10,.0f}  "
                  f"{p_mixed:>10,.0f}  {avg:>10,.0f}  {worst:>10,.0f}  {prior:>10,.0f}")

    _print_top("Top 10 by equal-weight MEAN across scenarios", 7, rows)
    _print_top("Top 10 by WORST-CASE (maximin)", 8, rows)
    _print_top("Top 10 by prior-weighted EV (0.25 low, 0.50 base, 0.25 high)", 9, rows)
    _print_top("Top 10 by MIXED (Discord-informed) scenario E[PnL]", 6, rows)


if __name__ == "__main__":
    n_teams = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(n_teams=n_teams, n_sims=n_sims)
