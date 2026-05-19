import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings

warnings.filterwarnings("ignore")

os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

TICKERS          = ["SPY", "QQQ", "BND", "VTI", "VWO", "VEA"]
START            = "2015-01-01"
END              = "2025-01-01"
INITIAL_CAPITAL  = 100_000
TRANSACTION_COST = 0.0005

SMA_FAST     = 50
SMA_SLOW     = 200
RSI_PERIOD   = 14
RSI_ENTRY    = 35
RSI_EXIT     = 70
TRADING_DAYS = 252

PALETTE = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw[tickers]
    raw.ffill(inplace=True)
    raw.dropna(how="all", inplace=True)
    return raw


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    diff = series.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def compute_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feats = {}
    for ticker in prices.columns:
        s = prices[ticker]
        feats[ticker] = pd.DataFrame({
            "close":    s,
            "sma_fast": s.rolling(SMA_FAST).mean(),
            "sma_slow": s.rolling(SMA_SLOW).mean(),
            "rsi":      _rsi(s, RSI_PERIOD),
        })
    return feats


def generate_signals(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = next(iter(feats.values())).index
    sig = pd.DataFrame(0.0, index=idx, columns=list(feats.keys()))

    for ticker, df in feats.items():
        uptrend         = df["sma_fast"] > df["sma_slow"]
        not_overbought  = df["rsi"] <= RSI_EXIT
        deeply_oversold = df["rsi"] < RSI_ENTRY

        sig[ticker] = np.where(
            uptrend & not_overbought, 1.0,
            np.where(~uptrend & deeply_oversold, 0.5, 0.0)
        )

    warmup = SMA_SLOW + RSI_PERIOD
    sig.iloc[:warmup] = 0.0
    return sig


def backtest_portfolio(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    daily_ret = prices.pct_change()

    sig_sum = signals.sum(axis=1).replace(0, np.nan)
    weights = signals.div(sig_sum, axis=0).fillna(0)

    w_prev   = weights.shift(1).fillna(0)
    turnover = w_prev.diff().abs().sum(axis=1).fillna(0)
    cost     = turnover * TRANSACTION_COST

    port_ret = (w_prev * daily_ret).sum(axis=1) - cost
    equity   = (1 + port_ret).cumprod() * INITIAL_CAPITAL
    equity.iloc[0] = INITIAL_CAPITAL
    return port_ret, equity


def buy_and_hold(prices: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series]:
    ret    = prices[ticker].pct_change().fillna(0)
    equity = (1 + ret).cumprod() * INITIAL_CAPITAL
    equity.iloc[0] = INITIAL_CAPITAL
    return ret, equity


def compute_metrics(returns: pd.Series, equity: pd.Series, label: str) -> dict:
    r       = returns.dropna()
    n_years = len(r) / TRADING_DAYS
    total   = equity.iloc[-1] / equity.iloc[0] - 1
    cagr    = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    vol     = r.std() * np.sqrt(TRADING_DAYS)
    sharpe  = r.mean() * TRADING_DAYS / vol if vol else 0.0

    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = cagr / abs(max_dd) if max_dd else 0.0
    win_rate = (r > 0).sum() / r.count()

    return {
        "Strategy":        label,
        "Total Return":    f"{total:.1%}",
        "CAGR":            f"{cagr:.1%}",
        "Ann. Volatility": f"{vol:.1%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_dd:.1%}",
        "Calmar Ratio":    f"{calmar:.2f}",
        "Win Rate (days)": f"{win_rate:.1%}",
    }


def _pct(x, _):
    return f"{x:.0f}%"


def plot_equity_curves(equities: dict[str, pd.Series], port_returns: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(3, 1, hspace=0.35, height_ratios=[3, 1.5, 1.5])

    ax1 = fig.add_subplot(gs[0])
    for (label, eq), color in zip(equities.items(), PALETTE):
        ax1.plot(eq.index, eq / eq.iloc[0] * 100, label=label,
                 color=color, linewidth=1.7)
    ax1.set_title("Portfolio vs Benchmarks  (2015 – 2025)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Growth of $100")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for (label, eq), color in zip(equities.items(), PALETTE):
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(eq.index, dd, 0, alpha=0.35, color=color, label=label)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_title("Drawdown", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    roll_sharpe = (
        port_returns.rolling(TRADING_DAYS).mean() /
        port_returns.rolling(TRADING_DAYS).std()
    ) * np.sqrt(TRADING_DAYS)
    ax3.plot(roll_sharpe.index, roll_sharpe, color=PALETTE[0], linewidth=1.3)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.6,
                label="Sharpe = 1")
    ax3.set_title("Rolling 1-Year Sharpe Ratio (Strategy)", fontsize=11)
    ax3.set_ylabel("Sharpe")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    fig.savefig("figures/equity_curves.png", dpi=150, bbox_inches="tight")
    return fig


def plot_signals_chart(
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    ticker: str = "SPY",
) -> plt.Figure:
    f   = feats[ticker]
    sig = signals[ticker]

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1], "hspace": 0.12},
    )

    ax1 = axes[0]
    ax1.plot(f.index, f["close"],    color="black",   linewidth=1,   label="Close")
    ax1.plot(f.index, f["sma_fast"], color="#1976D2", linewidth=1.4, label=f"SMA{SMA_FAST}")
    ax1.plot(f.index, f["sma_slow"], color="#E64A19", linewidth=1.4, label=f"SMA{SMA_SLOW}")
    ax1.fill_between(f.index,
                     f["close"].min(), f["close"].max(),
                     where=(f["sma_fast"] > f["sma_slow"]),
                     alpha=0.07, color="green", label="Bullish regime")
    ax1.set_title(f"{ticker}: Price & Moving Averages", fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.plot(f.index, f["rsi"], color="#7B1FA2", linewidth=1.1)
    ax2.axhline(RSI_EXIT,  color="red",   linestyle="--", alpha=0.7,
                label=f"Overbought ({RSI_EXIT})")
    ax2.axhline(RSI_ENTRY, color="green", linestyle="--", alpha=0.7,
                label=f"Oversold ({RSI_ENTRY})")
    ax2.axhline(50, color="gray", linewidth=0.5)
    ax2.fill_between(f.index, f["rsi"], RSI_ENTRY,
                     where=(f["rsi"] < RSI_ENTRY), alpha=0.25, color="green")
    ax2.fill_between(f.index, f["rsi"], RSI_EXIT,
                     where=(f["rsi"] > RSI_EXIT),  alpha=0.25, color="red")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI({RSI_PERIOD})")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25)

    ax3 = axes[2]
    ax3.fill_between(sig.index, sig, 0, step="post",
                     color="#1976D2", alpha=0.7, label="Position signal")
    ax3.set_ylim(-0.05, 1.15)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(["0 (flat)", "0.5 (half)", "1.0 (full)"])
    ax3.set_ylabel("Signal")
    ax3.set_xlabel("Date")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    fig.savefig(f"figures/{ticker.lower()}_signals.png", dpi=150, bbox_inches="tight")
    return fig


def plot_signal_heatmap(signals: pd.DataFrame) -> plt.Figure:
    try:
        monthly = signals.resample("ME").mean()
    except ValueError:
        monthly = signals.resample("M").mean()

    fig, ax = plt.subplots(figsize=(15, 3.5))
    im = ax.imshow(monthly.T.values, aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns)

    yr    = monthly.index.year.values
    ticks = [0] + [i for i in range(1, len(yr)) if yr[i] != yr[i - 1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(yr[i]) for i in ticks], rotation=45)

    plt.colorbar(im, ax=ax, label="Signal  (0 = out,  0.5 = half,  1 = full)")
    ax.set_title("Monthly Signal Strength by Asset", fontweight="bold")
    plt.tight_layout()
    fig.savefig("figures/signal_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


def plot_annual_returns(port_ret: pd.Series, spy_ret: pd.Series) -> plt.Figure:
    try:
        ap  = (1 + port_ret).resample("YE").prod() - 1
        as_ = (1 + spy_ret).resample("YE").prod() - 1
    except ValueError:
        ap  = (1 + port_ret).resample("A").prod() - 1
        as_ = (1 + spy_ret).resample("A").prod() - 1

    years = ap.index.year
    x, w  = np.arange(len(years)), 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, ap.values  * 100, w, label="Our Strategy",
           color=PALETTE[0], alpha=0.85)
    ax.bar(x + w / 2, as_.values * 100, w, label="SPY (B&H)",
           color=PALETTE[1], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title("Annual Returns: Strategy vs SPY (B&H)", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig("figures/annual_returns.png", dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(prices: pd.DataFrame) -> plt.Figure:
    corr = prices.pct_change().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Asset Return Correlations (2015–2025)", fontweight="bold")
    plt.tight_layout()
    fig.savefig("figures/correlation_matrix.png", dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    sep = "═" * 62
    print(sep)
    print("  ALGORITHMIC TRADING BACKTEST")
    print(f"  Assets   : {', '.join(TICKERS)}")
    print(f"  Period   : {START}  →  {END}")
    print(f"  Strategy : Momentum SMA{SMA_FAST}/{SMA_SLOW}  +  Mean-Reversion RSI{RSI_PERIOD}")
    print(sep)

    print("\n[1/6]  Downloading price data …")
    prices = download_prices(TICKERS, START, END)
    prices.to_csv("data/prices.csv")
    print(f"       {len(prices)} trading days, {len(prices.columns)} assets  →  data/prices.csv")

    print("\n[2/6]  Computing indicators (SMA, RSI) …")
    feats = compute_features(prices)

    print("\n[3/6]  Generating signals …")
    signals = generate_signals(feats)
    signals.to_csv("data/signals.csv")
    pct_active = (signals > 0).values.mean() * 100
    print(f"       Avg % of asset-days in market: {pct_active:.1f}%  →  data/signals.csv")

    print("\n[4/6]  Backtesting …")
    port_ret, port_eq = backtest_portfolio(prices, signals)
    spy_ret,  spy_eq  = buy_and_hold(prices, "SPY")
    qqq_ret,  qqq_eq  = buy_and_hold(prices, "QQQ")
    bnd_ret,  bnd_eq  = buy_and_hold(prices, "BND")

    print("\n[5/6]  Computing performance metrics …")
    results = [
        compute_metrics(port_ret, port_eq, "Our Strategy"),
        compute_metrics(spy_ret,  spy_eq,  "SPY  (B&H)"),
        compute_metrics(qqq_ret,  qqq_eq,  "QQQ  (B&H)"),
        compute_metrics(bnd_ret,  bnd_eq,  "BND  (B&H)"),
    ]
    df = pd.DataFrame(results).set_index("Strategy")
    df.to_csv("data/metrics.csv")

    print("\n" + sep)
    print("  PERFORMANCE SUMMARY")
    print(sep)
    print(df.to_string())
    print(sep)
    print("\n  Metrics saved →  data/metrics.csv")

    print("\n[6/6]  Generating plots …")
    equities = {
        "Our Strategy": port_eq,
        "SPY (B&H)":    spy_eq,
        "QQQ (B&H)":    qqq_eq,
        "BND (B&H)":    bnd_eq,
    }
    plot_equity_curves(equities, port_ret)
    print("       figures/equity_curves.png")

    plot_signals_chart(prices, feats, signals, "SPY")
    print("       figures/spy_signals.png")

    plot_signal_heatmap(signals)
    print("       figures/signal_heatmap.png")

    plot_annual_returns(port_ret, spy_ret)
    print("       figures/annual_returns.png")

    plot_correlation_matrix(prices)
    print("       figures/correlation_matrix.png")

    plt.show()
    print("\nAll outputs saved to  data/  and  figures/")
    print(sep)


if __name__ == "__main__":
    main()
