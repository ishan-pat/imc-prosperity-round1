# Round 1 — Notes

## Products

| Product | Strategy |
|---|---|
| `INTARIAN_PEPPER_ROOT` | Trend-long : holds max position (80) throughout the day |
| `ASH_COATED_OSMIUM`    | Passive multi-level market making with inventory skew and AC signal |

## Strategy

### INTARIAN_PEPPER_ROOT — Trend Long

Follows a deterministic linear trend: `fair_value = base_price + timestamp / 1000`, with the base price jumping +1000 each new competition day.

- Sweeps all asks within 10 ticks of fair value to build position
- Posts a passive bid at `min(best_bid + 1, int(fair_value) + 1)` for remaining capacity
- Never sells — holds max long (80 units) through the full day to capture the trend

### ASH_COATED_OSMIUM — Passive Market Making

Mean-reverting around 10,000 with a lag-1 return autocorrelation of −0.49.

- **4-level passive quoting** spread around a reservation price
- **Inventory skew**: `reservation = 10000 − position × (3/80)` — shifts quotes to reduce exposure
- **AC signal**: tightens bids after dips (`last_return < −3`), tightens asks after spikes (`last_return > +3`)
- **Strong AC**: shifts reservation ±1 when `|last_return| > 5`
- **End-of-day flattening**: progressively tightens quotes after `timestamp > 950,000` to close inventory

## Backtest Results (historical data)

```
  Day Product                        Pos         Cash          MtM          PnL
--------------------------------------------------------------------------------
   -2 INTARIAN_PEPPER_ROOT            80      -800528       880120        79592
   -2 ASH_COATED_OSMIUM               80      -791076       799480         8404
   -1 INTARIAN_PEPPER_ROOT            80            0       959840       959840
   -1 ASH_COATED_OSMIUM              -26      1071154      -260052       811102
    0 INTARIAN_PEPPER_ROOT            80            0      1040000      1040000
    0 ASH_COATED_OSMIUM              -80       549917      -800560      -250643
```

> Conservative estimate — only aggressive (taker) fills are simulated. Passive fill income is not counted.

## Lessons / What to carry into Round 2

_(fill in after Round 1 results are final — e.g. how often position limits bound, adverse selection on OSMIUM day 0, any missed signals)_
