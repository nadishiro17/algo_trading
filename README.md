# Assignment 3 — Algorithmic Trading with Backtesting

## Overview

Automated trading strategy applied to a diversified ETF portfolio.
The strategy combines **momentum** (SMA 50/200 golden-cross) with
**mean-reversion** (RSI-14 oversold/overbought) signals, backtested
over 2015–2025 and evaluated against three buy-and-hold benchmarks.

## Assets

| Ticker | Description |
|--------|-------------|
| SPY    | S&P 500 ETF (primary benchmark) |
| QQQ    | NASDAQ-100 ETF (benchmark) |
| BND    | US Aggregate Bond ETF (benchmark) |
| VTI    | US Total Stock Market |
| VWO    | Emerging Markets |
| VEA    | Developed Markets ex-US |

## Strategy: Momentum + Mean-Reversion Combo

**Signal per asset (daily):**

| Signal | Condition | Position size |
|--------|-----------|---------------|
| Full long | SMA50 > SMA200 AND RSI ≤ 70 | 1.0 |
| Mean-reversion catch | SMA50 ≤ SMA200 AND RSI < 35 | 0.5 |
| Flat / cash | otherwise | 0.0 |

**Portfolio construction:**
- Weight = signal_strength_i / Σ(all signal strengths) — strength-weighted equal allocation
- Rebalanced daily; signals shifted 1 day to prevent look-ahead bias
- Transaction cost: 0.05% per unit of turnover

**Key parameters:**

```
SMA_FAST  = 50       # fast moving average window
SMA_SLOW  = 200      # slow moving average window (trend filter)
RSI_PERIOD = 14      # RSI lookback
RSI_ENTRY  = 35      # oversold → mean-reversion buy signal
RSI_EXIT   = 70      # overbought → exit / no new entry
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| CAGR | Compound annual growth rate |
| Sharpe Ratio | Annualized return / annualized volatility |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | CAGR / |Max Drawdown| |
| Win Rate | % of days with positive returns |

## Repository Structure

```
assignment3/
├── algo_trading.py       ← main script
├── data/
│   ├── prices.csv        ← downloaded price data (generated on run)
│   ├── signals.csv       ← daily signals (generated on run)
│   └── metrics.csv       ← performance table (generated on run)
├── figures/
│   ├── equity_curves.png
│   ├── spy_signals.png
│   ├── signal_heatmap.png
│   ├── annual_returns.png
│   └── correlation_matrix.png
└── README.md
```

## How to Run

### Requirements

```bash
pip install yfinance pandas numpy matplotlib
```

### Run the backtest

```bash
cd FIN_ENG/assignment3
python algo_trading.py
```

The script will:
1. Download price data from Yahoo Finance (internet required)
2. Compute SMA and RSI indicators for all 6 assets
3. Generate daily signals
4. Run the portfolio backtest with transaction costs
5. Compare against SPY, QQQ, BND buy-and-hold benchmarks
6. Print a performance summary table to the console
7. Save 5 plots to `figures/` and 3 CSV files to `data/`

### Expected console output

```
══════════════════════════════════════════════════════════════
  ALGORITHMIC TRADING BACKTEST
  Assets   : SPY, QQQ, BND, VTI, VWO, VEA
  Period   : 2015-01-01  →  2025-01-01
  Strategy : Momentum SMA50/200  +  Mean-Reversion RSI14
══════════════════════════════════════════════════════════════
...
  PERFORMANCE SUMMARY
══════════════════════════════════════════════════════════════
                  Total Return  CAGR  Ann. Volatility  Sharpe ...
Strategy
Our Strategy         ...
SPY  (B&H)           ...
QQQ  (B&H)           ...
BND  (B&H)           ...
```
