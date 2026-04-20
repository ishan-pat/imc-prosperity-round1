# IMC Prosperity 4

Algorithmic and manual challenge submissions for IMC Prosperity 4.

## Layout

```
.
├── submissions/       # Trader code, one subfolder per round; `final.py` = submitted version
├── data/              # Raw data capsules from each round (CSVs gitignored — see data/README.md)
├── notebooks/         # EDA, plotting, ad-hoc analysis
├── backtesting/       # Local backtest harnesses, parameter sweeps, version comparisons
├── manual/            # Manual challenge scripts per round
├── writeups/          # Per-round notes, strategy rationale, post-round lessons
├── datamodel.py       # IMC-provided types (shared by all rounds)
└── requirements.txt
```

## Per-round pointers

| Round | Submission | Backtest | Notes |
|---|---|---|---|
| 1 | [submissions/round1/final.py](submissions/round1/final.py) | [backtesting/round1_backtest.py](backtesting/round1_backtest.py) | [writeups/round1_notes.md](writeups/round1_notes.md) |

## Running locally

**Requirements:** Python 3.10+, `pip install -r requirements.txt`

```bash
# Tests for the Round 1 trader
python3 -m pytest submissions/round1 -v

# Backtest (requires CSVs dropped into data/round1/ — see data/README.md)
python3 -m backtesting.round1_backtest
```

## Conventions

- **Submissions**: within `submissions/roundN/`, iterate as `trader_v1.py`, `trader_v2.py`, and copy the actually-submitted version to `final.py`.
- **Only one file is uploaded to the competition**: the chosen `final.py` (renamed to `trader.py` on upload). Everything else is local tooling.
- **Data**: CSVs / Parquet / JSON are gitignored. Keep the `data/roundN/` folders in git via `.gitkeep`.
- **Writeups**: capture non-obvious strategy rationale and post-round lessons per round so Round N+1 planning has context.
