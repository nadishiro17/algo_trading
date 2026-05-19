import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

BLUE       = "#1976D2"
DARK       = "#212121"
MID_GRAY   = "#616161"
LIGHT_GRAY = "#F5F5F5"
FOOTER_TXT = "Assignment 3 — Algorithmic Trading  |  Northwestern MSDS  |  2025"
REPORT_OUT = "report.pdf"


def _header(fig, title):
    ax = fig.add_axes([0, 0.945, 1, 0.055])
    ax.set_facecolor(BLUE)
    ax.text(0.5, 0.5, title, color="white", fontsize=14,
            fontweight="bold", ha="center", va="center")
    ax.axis("off")


def _footer(fig):
    ax = fig.add_axes([0, 0, 1, 0.028])
    ax.set_facecolor(LIGHT_GRAY)
    ax.text(0.5, 0.5, FOOTER_TXT, color=MID_GRAY,
            fontsize=7, ha="center", va="center")
    ax.axis("off")


def title_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    _footer(fig)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.88])
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch(
        (0, 0.72), 1, 0.22, boxstyle="round,pad=0.02",
        facecolor=BLUE, edgecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.87, "Algorithmic Trading", color="white",
            fontsize=22, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.79, "Momentum + Mean-Reversion Strategy  |  Backtesting Report",
            color="#BBDEFB", fontsize=11, ha="center", va="center",
            transform=ax.transAxes)

    meta = [
        ("Portfolio",  "SPY · QQQ · BND · VTI · VWO · VEA"),
        ("Period",     "January 2015 – January 2025"),
        ("Strategy",   "SMA 50/200 Golden Cross  +  RSI(14) Mean-Reversion"),
        ("Capital",    "$100,000 initial  |  0.05% transaction cost"),
        ("Course",     "Financial Engineering  —  Assignment 3"),
    ]

    y = 0.63
    for label, value in meta:
        ax.text(0.05, y, label + ":", fontsize=10, fontweight="bold",
                color=BLUE, va="top", transform=ax.transAxes)
        ax.text(0.30, y, value, fontsize=10, color=DARK,
                va="top", transform=ax.transAxes)
        ax.plot([0.02, 0.98], [y - 0.025, y - 0.025],
                color="#E0E0E0", linewidth=0.5, transform=ax.transAxes)
        y -= 0.07

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def text_page(pdf, header, items):
    """
    items: list of (kind, text)
      kind = 'h1' | 'h2' | 'body' | 'bullet' | 'space' | 'note'
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    _header(fig, header)
    _footer(fig)

    ax = fig.add_axes([0.08, 0.04, 0.84, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.96

    for kind, text in items:
        if y < 0.04:
            break

        if kind == "space":
            y -= float(text) if text else 0.02

        elif kind == "h1":
            ax.text(0, y, text, fontsize=12, fontweight="bold",
                    color=BLUE, va="top", transform=ax.transAxes)
            y -= 0.032
            ax.plot([0, 1], [y + 0.004, y + 0.004],
                    color=BLUE, linewidth=0.9, transform=ax.transAxes)
            y -= 0.018

        elif kind == "h2":
            ax.text(0, y, text, fontsize=10.5, fontweight="bold",
                    color=DARK, va="top", transform=ax.transAxes)
            y -= 0.035

        elif kind == "body":
            for line in textwrap.fill(text, width=95).split("\n"):
                ax.text(0, y, line, fontsize=9, color=DARK,
                        va="top", transform=ax.transAxes)
                y -= 0.030
            y -= 0.006

        elif kind == "bullet":
            first = True
            for line in textwrap.fill(text, width=90).split("\n"):
                ax.text(0.02, y, "•" if first else " ", fontsize=9,
                        color=BLUE, va="top", transform=ax.transAxes)
                ax.text(0.06, y, line, fontsize=9, color=DARK,
                        va="top", transform=ax.transAxes)
                y -= 0.030
                first = False
            y -= 0.004

        elif kind == "note":
            ax.add_patch(patches.FancyBboxPatch(
                (0, y - 0.06), 1, 0.07,
                boxstyle="round,pad=0.008",
                facecolor="#E3F2FD", edgecolor=BLUE,
                linewidth=0.6, transform=ax.transAxes))
            ax.text(0.02, y - 0.008, text, fontsize=8.5, color="#0D47A1",
                    va="top", style="italic", transform=ax.transAxes)
            y -= 0.075

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_page(pdf, header, img_path, caption=""):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    _header(fig, header)
    _footer(fig)

    top    = 0.04 if not caption else 0.07
    height = 0.85 if not caption else 0.82
    img_ax = fig.add_axes([0.04, top + (0.03 if caption else 0), 0.92, height])

    if os.path.exists(img_path):
        img_ax.imshow(mpimg.imread(img_path))
    else:
        img_ax.text(0.5, 0.5, f"[Missing: {img_path}]",
                    ha="center", va="center", color="red")
    img_ax.axis("off")

    if caption:
        cap = fig.add_axes([0.04, 0.034, 0.92, 0.035])
        cap.text(0.5, 0.5, caption, ha="center", va="center",
                 fontsize=8, color=MID_GRAY, style="italic")
        cap.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def metrics_table_page(pdf, csv_path):
    df = pd.read_csv(csv_path).set_index("Strategy")

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    _header(fig, "Performance Summary")
    _footer(fig)

    ax = fig.add_axes([0.03, 0.35, 0.94, 0.56])
    ax.axis("off")

    cols   = df.columns.tolist()
    rows   = df.index.tolist()
    data   = df.values.tolist()

    col_widths = [0.22] + [0.13] * len(cols)
    x_starts   = [sum(col_widths[:i]) for i in range(len(col_widths) + 1)]

    def cell(ax, x, y, w, h, text, bg, fg, fs=8.5, bold=False, ha="center"):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=bg,
                                       edgecolor="white", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center",
                fontsize=fs, color=fg,
                fontweight="bold" if bold else "normal")

    ROW_H   = 0.09
    HEADER_H = 0.10
    n_rows  = len(rows)
    total_h = HEADER_H + n_rows * ROW_H
    y_top   = total_h

    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h + 0.02)

    # Column headers
    cell(ax, x_starts[0], y_top, col_widths[0], HEADER_H,
         "Strategy", BLUE, "white", fs=9, bold=True)
    for j, col in enumerate(cols):
        cell(ax, x_starts[j + 1], y_top, col_widths[j + 1], HEADER_H,
             col, BLUE, "white", fs=8, bold=True)

    highlight_row = 0

    for i, (row_label, row_data) in enumerate(zip(rows, data)):
        y = y_top - HEADER_H - (i + 1) * ROW_H
        is_ours = (i == highlight_row)
        bg_row  = "#E3F2FD" if is_ours else ("#FAFAFA" if i % 2 == 0 else "white")
        fg_row  = BLUE if is_ours else DARK
        fw      = "bold" if is_ours else "normal"

        cell(ax, x_starts[0], y, col_widths[0], ROW_H,
             row_label, bg_row, fg_row, fs=8.5, bold=is_ours, ha="center")
        for j, val in enumerate(row_data):
            cell(ax, x_starts[j + 1], y, col_widths[j + 1], ROW_H,
                 str(val), bg_row, fg_row, fs=8.5, bold=is_ours)

    # Annotation box below table
    ann = fig.add_axes([0.06, 0.18, 0.88, 0.14])
    ann.axis("off")
    ann.add_patch(patches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.04",
        facecolor="#E3F2FD", edgecolor=BLUE, linewidth=0.8))
    notes = [
        "Our Strategy is highlighted in blue. Key observations:",
        "  • Volatility reduced from 17.6% (SPY) to 13.6% — strategy spends ~37% of time in cash.",
        "  • Max Drawdown of -29.7% is lower than both SPY (-33.7%) and QQQ (-35.1%).",
        "  • Sharpe ratio of 0.63 exceeds BND (0.28) but trails equity benchmarks.",
        "  • Strategy in market 63.2% of asset-days, avoiding worst drawdown periods.",
    ]
    for k, note in enumerate(notes):
        ann.text(0.02, 0.88 - k * 0.19, note, fontsize=8.5,
                 color="#0D47A1", va="top",
                 fontweight="bold" if k == 0 else "normal")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with PdfPages(REPORT_OUT) as pdf:

        title_page(pdf)

        text_page(pdf, "Abstract", [
            ("body",
             "This report presents a systematic algorithmic trading strategy that combines "
             "momentum and mean-reversion signals, backtested over a 10-year period (2015–2025) "
             "on a diversified portfolio of six ETFs. The strategy employs a 50/200-day Simple "
             "Moving Average crossover as a momentum filter and a 14-period Relative Strength "
             "Index (RSI) for mean-reversion entry and exit timing."),
            ("space", "0.02"),
            ("body",
             "Over the backtest period, the strategy achieved a Compound Annual Growth Rate (CAGR) "
             "of 7.9% with annualized volatility of 13.6% and a Sharpe ratio of 0.63. The strategy "
             "underperformed equity benchmarks (SPY: 13.0% CAGR, QQQ: 18.3% CAGR) in absolute "
             "return during the predominantly bullish decade, which is expected for any timing-based "
             "approach. However, it reduced maximum drawdown to -29.7% compared to -33.7% for SPY "
             "and -35.1% for QQQ, demonstrating meaningful downside protection."),
            ("space", "0.03"),
            ("h1", "1. Problem Description"),
            ("body",
             "The objective of this assignment is to design, implement, and backtest a fully "
             "automated algorithmic trading strategy applied to a multi-asset ETF portfolio. "
             "The portfolio spans six exchange-traded funds covering distinct market segments:"),
            ("bullet", "SPY — S&P 500 large-cap U.S. equities (primary benchmark)"),
            ("bullet", "QQQ — NASDAQ-100 technology-heavy U.S. equities (benchmark)"),
            ("bullet", "BND — U.S. aggregate bond fund (fixed-income benchmark)"),
            ("bullet", "VTI — U.S. total stock market"),
            ("bullet", "VEA — Developed international markets (ex-U.S.)"),
            ("bullet", "VWO — Emerging markets equities"),
            ("space", "0.02"),
            ("body",
             "The core challenge is to construct a trading rule that adapts to different market "
             "regimes — trending markets where momentum strategies excel and mean-reverting "
             "markets where oversold bounces provide alpha — and to rigorously assess its "
             "risk-adjusted performance against passive buy-and-hold benchmarks using the "
             "Sharpe ratio, maximum drawdown, CAGR, and Calmar ratio."),
            ("space", "0.02"),
            ("note",
             "Reference: Clenow, A. (2019). Trading Evolved. The strategy design follows "
             "Clenow's principles of systematic, rules-based trend following applied to "
             "diversified ETF universes."),
        ])

        text_page(pdf, "Data Preparation & Pipeline", [
            ("h1", "2. Data Preparation and Pipeline"),
            ("body",
             "Historical adjusted-close price data for all six assets was obtained from Yahoo "
             "Finance via the yfinance Python library, covering January 1, 2015, to January 1, "
             "2025 — a total of 2,516 trading days. Adjusted prices account for dividends and "
             "stock splits, ensuring accurate return calculations over the full period."),
            ("space", "0.015"),
            ("h2", "Data Pipeline Steps"),
            ("bullet",
             "Download adjusted close prices for all tickers using yfinance (auto_adjust=True)."),
            ("bullet",
             "Forward-fill missing values to handle market holidays and non-trading days "
             "consistently across all assets."),
            ("bullet",
             "Drop rows that are fully missing across all tickers."),
            ("bullet",
             "Compute technical indicators per asset: 50-day SMA, 200-day SMA, RSI(14)."),
            ("bullet",
             "Apply a warm-up period of 214 bars (200 + 14) before any signals are activated "
             "to prevent indicator artifacts."),
            ("bullet",
             "Save raw prices to data/prices.csv and daily signals to data/signals.csv."),
            ("space", "0.02"),
            ("h2", "Indicator Computation"),
            ("body",
             "Simple Moving Averages (SMA) are computed using a standard rolling window mean. "
             "The RSI uses Wilder's exponential smoothing method (EWM with com = period - 1) "
             "which matches the output of most charting platforms and trading terminals. "
             "This choice avoids the step-function artifacts of a simple rolling average "
             "in RSI calculation."),
            ("space", "0.02"),
            ("h1", "3. Research Design"),
            ("h2", "3.1 Strategy Architecture"),
            ("body",
             "The strategy combines two complementary alpha sources that tend to perform in "
             "different market environments:"),
            ("body",
             "Momentum Component — A golden cross / death cross system. When SMA50 crosses "
             "above SMA200, the asset is in a confirmed uptrend (bullish regime). This filter "
             "keeps the portfolio out of assets in sustained downtrends, reducing drawdown "
             "exposure. The SMA200 represents approximately 10 months of price history, "
             "filtering short-term noise while capturing medium-to-long term trends."),
            ("body",
             "Mean-Reversion Component — RSI(14) provides entry timing within trends and "
             "opportunistic counter-trend entries. The RSI prevents buying into overbought "
             "conditions (RSI > 70) and catches deeply oversold bounces (RSI < 35) even "
             "against the trend, sizing those at half position."),
            ("space", "0.02"),
            ("h2", "3.2 Signal Logic"),
            ("bullet",
             "FULL LONG (signal = 1.0): SMA50 > SMA200 AND RSI ≤ 70. Asset is in uptrend "
             "and not overbought. Standard momentum long."),
            ("bullet",
             "HALF LONG (signal = 0.5): SMA50 ≤ SMA200 AND RSI < 35. Asset is in downtrend "
             "but deeply oversold — mean-reversion catch with reduced size."),
            ("bullet",
             "FLAT / CASH (signal = 0.0): All other conditions. No exposure."),
            ("space", "0.02"),
            ("h2", "3.3 Portfolio Construction"),
            ("body",
             "Weights are assigned proportionally to signal strength each day: "
             "weight_i = signal_i / Σ(all signals). This ensures the portfolio is fully "
             "invested (when any signal is active) while giving 2× weight to full-signal "
             "assets versus half-signal assets. Signals are shifted 1 day forward before "
             "computing returns to prevent look-ahead bias. A transaction cost of 0.05% per "
             "unit of portfolio turnover is applied daily to reflect real-world trading friction."),
        ])

        metrics_table_page(pdf, "data/metrics.csv")

        figure_page(
            pdf,
            "Equity Curves, Drawdowns & Rolling Sharpe",
            "figures/equity_curves.png",
            caption=(
                "Figure 1: Top — normalized growth of $100 for strategy vs benchmarks (2015–2025).  "
                "Middle — drawdown from peak for each series.  "
                "Bottom — rolling 1-year Sharpe ratio of the strategy."
            ),
        )

        figure_page(
            pdf,
            "SPY: Price Signals & RSI",
            "figures/spy_signals.png",
            caption=(
                "Figure 2: SPY price with SMA50 (blue) and SMA200 (red). Green shading = bullish regime. "
                "Middle panel: RSI(14) with overbought/oversold bands. "
                "Bottom panel: position signal (0, 0.5, or 1.0) for SPY over time."
            ),
        )

        figure_page(
            pdf,
            "Annual Returns & Signal Heatmap",
            "figures/annual_returns.png",
            caption="Figure 3: Year-by-year returns of the strategy vs SPY buy-and-hold.",
        )

        figure_page(
            pdf,
            "Monthly Signal Heatmap",
            "figures/signal_heatmap.png",
            caption=(
                "Figure 4: Monthly average signal strength per asset (green = fully invested, "
                "red = out of market). Shows how the strategy rotates across assets over time."
            ),
        )

        figure_page(
            pdf,
            "Asset Correlation Matrix",
            "figures/correlation_matrix.png",
            caption=(
                "Figure 5: Pearson correlation of daily returns across all six assets (2015–2025). "
                "Low correlation between BND/VWO and U.S. equities supports portfolio diversification."
            ),
        )

        text_page(pdf, "Results & Conclusions", [
            ("h1", "4. Backtesting Results"),
            ("body",
             "Over the 10-year backtest period the strategy was in the market approximately "
             "63.2% of asset-days, holding cash or reduced exposure during bearish regimes. "
             "The portfolio started at $100,000 and grew to approximately $214,500 — a total "
             "return of 114.5%. Key findings:"),
            ("bullet",
             "CAGR of 7.9% significantly exceeds the bond benchmark BND (1.4% CAGR) and "
             "represents solid real returns above long-run inflation."),
            ("bullet",
             "Volatility of 13.6% is meaningfully lower than both SPY (17.6%) and QQQ (21.8%), "
             "reflecting the value of timing-based exposure management. The strategy holds cash "
             "roughly 37% of the time, dampening both upside and downside moves."),
            ("bullet",
             "Sharpe ratio of 0.63 is positive and well above BND (0.28), confirming the "
             "strategy earns a meaningful risk premium. It trails SPY (0.78) and QQQ (0.88), "
             "primarily because 2015–2025 was an exceptionally persistent bull market in U.S. "
             "equities where holding cash was costly."),
            ("bullet",
             "Maximum drawdown of -29.7% is 4 percentage points better than SPY (-33.7%) and "
             "5.4 points better than QQQ (-35.1%), demonstrating that the momentum filter "
             "successfully reduced peak-to-trough losses during the COVID crash (2020) and "
             "the 2022 rate-hike correction."),
            ("bullet",
             "Win rate of 46.4% (fewer than half of trading days were profitable) is typical "
             "of trend-following strategies: many small losses are offset by larger gains "
             "when trends persist. This also reflects days spent in cash earning 0%."),
            ("space", "0.025"),
            ("h1", "5. Conclusions"),
            ("body",
             "The Momentum + Mean-Reversion combo strategy demonstrated moderate risk-adjusted "
             "performance over the 2015–2025 backtest period. While it did not outperform passive "
             "equity benchmarks in absolute or Sharpe terms during this persistently bullish decade, "
             "it achieved its primary objectives of reducing portfolio volatility, limiting maximum "
             "drawdown, and significantly outperforming the bond benchmark."),
            ("space", "0.015"),
            ("body",
             "The strategy's defensive properties would likely be more valuable during prolonged "
             "bear markets (e.g., 2000–2002 dot-com bust, 2008–2009 financial crisis) where "
             "buy-and-hold equity strategies suffered -50% to -57% drawdowns. The SMA200 filter "
             "would have kept the portfolio largely in cash or reduced positions during those "
             "sustained downtrends."),
            ("space", "0.015"),
            ("h2", "Future Improvements"),
            ("bullet",
             "Volatility targeting / risk parity: scale position sizes inversely to realized "
             "volatility so each asset contributes equal risk regardless of its historical vol."),
            ("bullet",
             "Sector rotation: rank assets by momentum score and hold only the top N, improving "
             "concentration in the strongest trends."),
            ("bullet",
             "Macro regime filter: incorporate a VIX or yield-curve signal to go to cash "
             "during high-stress periods regardless of individual asset signals."),
            ("bullet",
             "Walk-forward optimization: test strategy parameters (SMA windows, RSI thresholds) "
             "using rolling in-sample/out-of-sample splits to avoid overfitting."),
            ("space", "0.025"),
            ("note",
             "All code available in algo_trading.py. Run: python algo_trading.py — requires "
             "yfinance, pandas, numpy, matplotlib. Data files saved to data/, figures to figures/."),
        ])

    print(f"Report saved →  {REPORT_OUT}")


if __name__ == "__main__":
    main()
