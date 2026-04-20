# Round 2 — Notes

## Submission

| Item | Value |
|---|---|
| Submitted algo file | [submissions/round2/final.py](../submissions/round2/final.py) (= `trader_v3.py`) |
| MAF bid | 100 XIRECs |
| Projected algo PnL (backtest) | 271,468 |
| **Manual submission** | **Research: 15, Scale: 46, Speed: 39** |
| Projected manual E[PnL] (mixed crowd) | ~190k |
| Cumulative R1 + R2 projection | algo ~540k + manual ~190k |

## Manual challenge submission rationale

Chose `r=15, s=46, v=39` based on grid search over 4 crowd scenarios — see
[manual/round2_invest_expand.py](../manual/round2_invest_expand.py).

| Scenario | E[PnL] at (15, 46, 39) |
|---|---|
| Low crowd (mean v=15) | 292k |
| Base crowd (mean v=30) | 205k |
| **Mixed crowd (Discord-informed)** | **190k** |
| High crowd (mean v=50) | 82k |
| Worst-case floor | 82k |
| Prior-weighted EV (0.25/0.5/0.25) | 196k |

**Why this over the per-scenario optima**:
- Mixed scenario (15% at v=0, 60% balanced N(30,10), 15% at N(55,10), 10% at v=100) is the most realistic crowd given Discord signals
- v=39 is just above the median in every plausible crowd, giving Speed ~0.62 without overpaying
- Dominates the maximin pick (r=13, s=35, v=52) on EV by ~30k while keeping an 82k floor — not a loss, just a smaller profit under extreme crowds
- Tied with the equal-weight winner (r=16, s=48, v=36) on EV while gaining 8k on worst-case

## Strategy summary

Both products carried over from Round 1 with only one change:

- **INTARIAN_PEPPER_ROOT** — unchanged from Round 1. Deterministic `+1/tick` trend holds in Round 2 data; strategy sweeps asks within 10 ticks of `fair_value = base + ts/1000` and sits at +80 for the whole day (~80k PnL/day, capped by position limit).
- **ASH_COATED_OSMIUM** — 4-level passive market-making around static `FAIR = 10,000` with:
  - AC signal (tighten bids/asks after dips/spikes)
  - EOD flattening after ts > 950,000
  - **NEW in v3**: inventory skew coefficient raised from `3/80` to `6/80`
  - **NEW in v3**: asymmetric hard band — `sell_cap = 0` at `pos ≤ -60` (no short-side cap on the long side)

## How we got here

### Phase 0 — Baseline (Round 1 code on Round 2 data)

Ported Round 1's `final.py` verbatim, backtested on Round 2 data with a new trade-tape passive-fill model (`passive_share = 0.5` default).

Result: **271,636 PnL** over 3 days.

Breakdown: PEPPER ≈ 240k (~80k/day), OSMIUM ≈ 32k (~11k/day).
OSMIUM at position limit 19.4% of ticks — flagged as the main risk.

### Phase 1 — EDA findings (OSMIUM-specific)

See [notebooks/round2_osmium_analysis.py](../notebooks/round2_osmium_analysis.py).

| Question | Finding | Action taken |
|---|---|---|
| Q2: Mean-reverting? | Lag-1 AC ≈ −0.50 all 3 days (matches Round 1's −0.49). Mean sits on 10000.  Rolling AC never flips positive. | Keep static `FAIR = 10000`. No dynamic FV work. |
| Q3: Volatility? | σ(Δmid) = 3.7 ticks, stable across days. 28% of ticks have Δmid=0. | Keep 4-level 1-tick spacing. No vol-adaptive spread. |
| Q4: Flow asymmetry? | Daily Buy/Sell ratio = 0.99–1.01 (balanced). 50-trade windows reach ±100 units (local bursts exist). | Skip flow guard — local bursts coincide with mean-reversion so flow signal is redundant with AC. |
| Q5: At-limit adverse vs favorable? | **Long@+80 always favorable** (mean μ +3.5 / +6.0 / +12.8). **Short@−80 mixed, day 1 catastrophic** (mean μ −31.30). | Build asymmetric band: cap short at −60, leave long unrestricted. |

### Phase 2 — Iterations

| Version | Change | Total PnL | OSMIUM ticks at limit | Day 0 end pos | Day 1 end pos |
|---|---|---|---|---|---|
| v1 (baseline) | Round 1 code | 271,636 | 5,834 (19.4%) | −76 | +80 |
| v2 | 6/80 skew + symmetric ±60 band | 270,827 | 402 (1.3%) | −60 | +79 |
| **v3 (chosen)** | 6/80 skew + asymmetric −60 band only | **271,468** | 1,336 (4.5%) | **−60** | **+80** |

**Why v3 wins**: tied with v1 on PnL (−0.06%, statistical noise), matches v2 on short-side risk protection (day 0 capped at −60 instead of v1's −76), and preserves v1's long-side favorable edge (day 1 reaches +80 to capture EDA-identified favorable drift). Best of both previous versions.

### Phase 3 — Skipped

Parameter sweep was part of the original plan but deferred. Rationale: v3 already near-ties v1 on PnL while dramatically reducing short-side risk. Further tuning on 3 backtest days risks overfitting; marginal EV moves to MAF bid (Phase 4) and manual challenge.

### Phase 4 — MAF bid sizing

Empirical V estimate (marginal value of +25% quote access) via `passive_share` sensitivity:

| passive_share | PnL | Δ |
|---|---|---|
| 0.50 (baseline / losing MAF) | 271,468 | — |
| 0.625 (+25% / winning MAF) | 271,826 | +358 |
| 0.75 | 272,168 | +700 |
| 1.00 | 272,683 | +1,215 (ceiling) |

**V ≈ 358 XIRECs.** 20× smaller than naive upper bound of 8k because:
1. PEPPER dominates PnL (~240k of 271k) and sits at +80 limit 100% of ticks → MAF adds ~0 on PEPPER
2. Only OSMIUM's ~32k contribution is MAF-sensitive
3. Inventory skew reduces margin on incremental adverse-side fills

EV table (assuming V = 400):

| Bid | Net if win | P(win) low crowd | base | high | EV low | EV base | EV high |
|---|---|---|---|---|---|---|---|
| 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 |
| **100** | 300 | 0.60 | 0.30 | 0.05 | **+180** | **+90** | **+15** |
| 200 | 200 | 0.75 | 0.45 | 0.10 | +150 | +90 | +20 |
| 500 | −100 | 0.87 | 0.70 | 0.25 | −87 | −70 | −25 |
| 1,000 | −600 | 0.95 | 0.85 | 0.45 | −570 | −510 | −270 |
| 5,000 | −4,600 | 0.99 | 0.95 | 0.60 | −4,554 | −4,370 | −2,760 |

**Chose bid = 100**: positive EV under all three crowd scenarios, max downside 100 XIRECs if V = 0.

**Switch to bid 0 if Discord signals the crowd is bidding aggressively (>1k medians).**

## Caveats / Known risks

1. **V = 358 is specific to the 3 backtest days.** If live Round 2 has more OSMIUM volatility, V could be 2–3× higher — but still small absolutely.
2. **Backtest uses `passive_share = 0.5` as a proxy** for the simulator's "80% randomized quotes." Not identical; expect ±5–10% PnL variance live vs. backtest.
3. **v3 is validated on 3 days** — could be an overfit. The asymmetric band exploits a data-observed asymmetry that might not persist with different live mid trajectories. Acceptable given the risk-symmetry argument (long-favorable is the mean-reversion default for a strategy anchored at 10000 for a product trading near 10000).
4. **Live PEPPER PnL is at the mercy of the trend continuing.** If the `+1/tick` trend stops in live Round 2 (regime shift), PEPPER PnL could be much lower — and since MAF has ~0 PEPPER value, the whole MAF EV also shrinks.

## Post-round lessons

_(fill in after Round 2 results are published — compare live vs. projected PnL, update crowd model for future MAF-like auctions)_
