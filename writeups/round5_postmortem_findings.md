# Round 5 — post-round findings

After live results came in, re-ran the analysis at realistic fill model and
added leave-one-day-out cross-validation to harden future decisions.

## 1. Realistic fill model collapses the v1/v2 difference

Re-ran v1 and v2 with `passive_share=0.1` (default in
`backtesting/round5_attribution.py` was 0.5):

| | v1 train | v2 train | v1 validate | v2 validate |
|---|---:|---:|---:|---:|
| PnL @ p=0.5 | 166,142 | 143,090 | 78,048 | 79,740 |
| PnL @ p=0.1 | **143,020** | **143,090** | **80,000** | **79,740** |
| Diff @ p=0.5 | — | −14% | — | +2.2% |
| Diff @ p=0.1 | — | **0%** | — | **−0.3%** |

At realistic fill rates, **v1 and v2 are statistically indistinguishable**.
The entire $23K of v1's in-sample AR(1) PnL was a fill-model artifact. The
live $3-5K v1/v2 gap is consistent with noise, not a real cost of the AR(1)
disable decision.

**Implication for round 6+**: run all backtests with
`--passive-share 0.1`. The `0.5` default systematically promotes passive
market-making strategies by attributing 50% of trade volume to our quotes, an
edge that does not exist when many MMs compete.

## 2. Trend strategy is stable across all 3 days

Leave-one-day-out CV (`backtesting/round5_loo_cv.py`) at passive_share=0.1
shows every trend product is positive on every held-out day:

| Product | Day 2 | Day 3 | Day 4 | Min | Mean |
|---|---:|---:|---:|---:|---:|
| MICROCHIP_OVAL | 5,130 | 18,195 | 18,940 | +5,130 | +14,088 |
| PEBBLES_XS | 19,455 | 12,910 | 7,610 | +7,610 | +13,325 |
| OXYGEN_SHAKE_GARLIC | 18,225 | 1,035 | 19,510 | +1,035 | +12,923 |
| GALAXY_SOUNDS_BLACK_HOLES | 14,405 | 6,810 | 13,125 | +6,810 | +11,447 |
| UV_VISOR_AMBER | 14,935 | 11,030 | 2,500 | +2,500 | +9,488 |
| PANEL_2X4 | 7,340 | 7,330 | 8,895 | +7,330 | +7,855 |
| PEBBLES_S | 5,850 | 240 | 9,420 | +240 | +5,170 |

Mean per-day PnL across folds: $74,297 (v1), $74,277 (v2). Identical.

**Stability tiers** (lowest single-day PnL):

- **Robust** (min > $5K): PANEL_2X4, PEBBLES_XS, GALAXY_SOUNDS_BLACK_HOLES,
  MICROCHIP_OVAL.
- **Fragile** (min < $3K): UV_VISOR_AMBER (decelerating), OXYGEN_SHAKE_GARLIC
  (one-off weak day), PEBBLES_S (high variance).

No product is a candidate for outright disable — every one was positive
every day in-sample. Size reduction on the fragile tier is defensible but P1
already showed that drift-weighted sizing doesn't improve risk-adjusted
return; a manually tuned size cut for just UV_VISOR_AMBER + PEBBLES_S would
be the minimum-evidence variant.

## 3. The 5× backtest-vs-live gap persists at realistic passive_share

| | Backtest day 4 (p=0.1) | Live day 5 |
|---|---:|---:|
| v1 | 80,000 | ~18,000 |
| v2 | 79,740 | ~15,000 |

Still ~4.5× optimistic. Lowering passive_share alone doesn't close this — it
fixes the passive-fill side, but trend strategy is dominated by aggressive
crossing fills, which the simulator over-attributes by assuming we can sweep
entire book levels instantly without competition. Closing this gap would
require modeling MM competitor priority at price levels — out of scope.

**Workaround for round 6+**: when reasoning about live expectations from
backtest numbers, divide absolute backtest PnL by ~5×.

## 4. What would actually push this further

Given Round 5 is over, ranked by where the residual alpha would come from:

1. **Pairs / cointegration trading within categories** — completely untouched
   by current trader. Within MICROCHIP, PEBBLES, etc., products share
   fundamentals; spreads should mean-revert on minute-scale. Avellaneda-Lee
   2010 statistical arbitrage. The screening file already runs Engle-Granger
   per category in `notebooks/round5_screening.py` but the trader doesn't
   read it. This is the only orthogonal alpha source still on the table.

2. **Tighter kill-switch at realistic fill rate.** The −$10K threshold was
   set based on `passive_share=0.5` drawdowns. At 0.1 the worst-case MtM is
   shallower and the threshold could probably tighten to −$5K to −$7K
   without firing on backtest. Re-run the diagnostic from the v2 P5 step to
   confirm.

3. **Manually size-cut the fragile-tier products** (UV_VISOR_AMBER from 10
   to 6, PEBBLES_S from 10 to 6), keeping robust-tier at 10. The intuition
   from drift-weighted sizing was wrong, but per-day stability evidence is
   stronger than drift magnitude as a sizing signal.

For the next round, two procedural changes would prevent v2-style mistakes:

- Always use `passive_share=0.1` as the backtest default.
- Always run LOO-CV before binary disable/enable decisions.
- Require ≥20% backtest delta before shipping a change (the realistic noise
  floor given the 5× live discount factor).
