"""
visualize.py
============
All publication-quality charts for the Job Posting Velocity strategy.

Every function saves its figure to ``outputs/`` at 300 DPI and also
returns the ``matplotlib.figure.Figure`` object for use in notebooks.

Style: seaborn-whitegrid throughout, with a custom earthy-teal palette
that reads well in both light and dark README renderings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

matplotlib.use("Agg")  # Non-interactive backend — safe for headless environments

logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────────────
try:
    plt.style.use(config.FIGURE_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")

_PALETTE = {
    "long": "#2C7BB6",    # steel blue — bullish
    "short": "#D7191C",   # crimson — bearish
    "neutral": "#6B6B6B", # grey
    "beat": "#1A9641",    # forest green
    "miss": "#FDAE61",    # amber
    "accent": "#ABD9E9",  # light blue
}

_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_SQUARE = (8, 7)
_FIGSIZE_TALL = (10, 8)


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save figure to outputs/ directory at configured DPI."""
    path = config.OUTPUTS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    logger.info("Saved chart: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Posting timeseries
# ─────────────────────────────────────────────────────────────────────────────

def plot_posting_timeseries(
    timeseries_df: pd.DataFrame,
    ticker: str,
) -> plt.Figure:
    """
    Line chart of job posting count and net-new postings over time for one company.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        Full universe timeseries.  Must contain ``["ticker", "year_month",
        "posting_count", "net_new", "long_signal"]``.
    ticker : str
        Ticker to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/posting_timeseries_{ticker}.png``.
    """
    df = timeseries_df[timeseries_df["ticker"] == ticker].copy()
    if df.empty:
        logger.warning("No data for ticker %s", ticker)
        return plt.figure()

    df["period"] = pd.to_datetime(df["year_month"], format="%Y%m", errors="coerce")
    df = df.sort_values("period")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_FIGSIZE_WIDE, sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # ── Top panel: posting count ──────────────────────────────────────────
    ax1.fill_between(df["period"], df["posting_count"],
                     alpha=0.25, color=_PALETTE["accent"])
    ax1.plot(df["period"], df["posting_count"],
             color=_PALETTE["long"], linewidth=2, label="Job Postings")

    # Mark long-signal months
    if "long_signal" in df.columns:
        signal_rows = df[df["long_signal"] == True]
        for _, row in signal_rows.iterrows():
            ax1.axvline(row["period"], color=_PALETTE["long"],
                        alpha=0.4, linewidth=1.2, linestyle="--")

    ax1.set_ylabel("Open Postings", fontsize=10)
    ax1.set_title(f"{ticker} — Job Posting Timeseries", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    # ── Bottom panel: net-new ─────────────────────────────────────────────
    if "net_new" in df.columns:
        pos_mask = df["net_new"] >= 0
        ax2.bar(df.loc[pos_mask, "period"], df.loc[pos_mask, "net_new"],
                color=_PALETTE["beat"], alpha=0.7, width=20, label="Net New (positive)")
        ax2.bar(df.loc[~pos_mask, "period"], df.loc[~pos_mask, "net_new"],
                color=_PALETTE["miss"], alpha=0.7, width=20, label="Net New (negative)")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Month-over-Month Δ", fontsize=10)
        ax2.legend(fontsize=8)

    ax2.set_xlabel("Month", fontsize=10)
    fig.tight_layout()
    _save(fig, f"posting_timeseries_{ticker}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Functional decomposition heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_functional_decomposition_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap of signal_ratio across companies (y) and months (x).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``["ticker", "year_month", "signal_ratio"]``.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/functional_decomposition_heatmap.png``.

    Notes
    -----
    This is the most visually striking chart in the repository.
    Warm colours (red/orange) indicate low signal_ratio (overhead-dominated
    hiring); cool colours (blue/teal) indicate high signal_ratio (growth-
    function-dominated hiring).  The threshold line at ``config.SIGNAL_RATIO_THRESHOLD``
    is drawn as a text annotation.
    """
    pivot = df.pivot_table(
        index="ticker",
        columns="year_month",
        values="signal_ratio",
        aggfunc="mean",
    )

    # Sort tickers by mean signal_ratio descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Limit x-axis to last 24 months for readability
    if pivot.shape[1] > 24:
        pivot = pivot.iloc[:, -24:]

    fig, ax = plt.subplots(figsize=(max(12, pivot.shape[1] * 0.5),
                                    max(6, pivot.shape[0] * 0.55)))
    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        center=config.SIGNAL_RATIO_THRESHOLD,
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="#e0e0e0",
        annot=False,
        cbar_kws={"label": "Signal Ratio (Engineering+Product+Sales / Total)"},
    )

    ax.set_title(
        "Functional Hiring Decomposition — Signal Ratio by Company × Month",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Year-Month", fontsize=10)
    ax.set_ylabel("Company (Ticker)", fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    _save(fig, "functional_decomposition_heatmap.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Z-score distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_zscore_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Histogram of hiring_accel_zscore coloured by long_signal flag.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``["hiring_accel_zscore", "long_signal"]``.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/zscore_distribution.png``.
    """
    df = df.dropna(subset=["hiring_accel_zscore"])

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    long_z = df[df["long_signal"] == True]["hiring_accel_zscore"]
    other_z = df[df["long_signal"] != True]["hiring_accel_zscore"]

    bins = np.linspace(df["hiring_accel_zscore"].quantile(0.01),
                       df["hiring_accel_zscore"].quantile(0.99), 35)

    ax.hist(other_z, bins=bins, color=_PALETTE["neutral"],
            alpha=0.65, label="No Signal", edgecolor="white")
    ax.hist(long_z, bins=bins, color=_PALETTE["long"],
            alpha=0.75, label="Long Signal", edgecolor="white")

    ax.axvline(config.HIRING_ACCEL_ZSCORE_THRESHOLD, color=_PALETTE["short"],
               linestyle="--", linewidth=1.5,
               label=f"Threshold = {config.HIRING_ACCEL_ZSCORE_THRESHOLD}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Hiring Acceleration Z-Score", fontsize=11)
    ax.set_ylabel("Company-Month Count", fontsize=11)
    ax.set_title("Distribution of Hiring Acceleration Z-Scores\n"
                 "(coloured by Long Signal flag)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, "zscore_distribution.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Signal vs earnings surprise scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_vs_earnings_surprise(
    aligned_df: pd.DataFrame,
    ic_value: Optional[float] = None,
) -> plt.Figure:
    """
    Scatter of signal_strength (x) vs forward earnings surprise_pct (y).

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Must contain ``["signal_strength", "surprise_pct", "beat_flag"]``.
    ic_value : float, optional
        Pre-computed Spearman IC to annotate in the corner.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/signal_vs_surprise_scatter.png``.

    Notes
    -----
    OLS trendline is overlaid.  Upward slope (positive β) is consistent with
    the hypothesis that higher signal_strength predicts a larger positive
    earnings surprise.
    """
    df = aligned_df.dropna(subset=["signal_strength", "surprise_pct"]).copy()
    if df.empty:
        logger.warning("No data for signal vs surprise scatter")
        return plt.figure()

    # Clip extreme surprises for visual clarity
    clip_q = df["surprise_pct"].quantile([0.02, 0.98])
    df["surprise_clipped"] = df["surprise_pct"].clip(*clip_q)

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    beat_mask = df["beat_flag"] == True
    ax.scatter(
        df.loc[beat_mask, "signal_strength"],
        df.loc[beat_mask, "surprise_clipped"],
        c=_PALETTE["beat"], alpha=0.65, s=55, label="Beat", edgecolors="white",
        linewidth=0.4, zorder=3,
    )
    ax.scatter(
        df.loc[~beat_mask, "signal_strength"],
        df.loc[~beat_mask, "surprise_clipped"],
        c=_PALETTE["miss"], alpha=0.65, s=55, label="Miss", edgecolors="white",
        linewidth=0.4, zorder=3,
    )

    # OLS trendline
    x = df["signal_strength"].values
    y = df["surprise_clipped"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line,
            color="black", linewidth=1.5, linestyle="--",
            label=f"OLS (β={slope:.2f}, p={p:.3f})")

    ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.7, linestyle=":")

    ax.set_xlabel("Signal Strength (Z-Score × Signal Ratio)", fontsize=11)
    ax.set_ylabel("Forward EPS Surprise %", fontsize=11)
    ax.set_title("Signal Strength vs. Forward Earnings Surprise",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    if ic_value is not None:
        ax.text(0.04, 0.96, f"IC (Spearman) = {ic_value:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))

    fig.tight_layout()
    _save(fig, "signal_vs_surprise_scatter.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cumulative PnL
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_pnl(backtest_results_df: pd.DataFrame) -> plt.Figure:
    """
    Cumulative long-short PnL over time with shaded drawdown regions.

    Parameters
    ----------
    backtest_results_df : pd.DataFrame
        Output of :func:`src.backtest.run_long_short_backtest`.
        Must contain ``["month", "cumulative_pnl", "ls_return"]``.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/cumulative_pnl.png``.
    """
    df = backtest_results_df.dropna(subset=["cumulative_pnl"]).copy()
    if df.empty:
        logger.warning("No data for PnL chart")
        return plt.figure()

    df["period"] = pd.to_datetime(df["month"], format="%Y%m", errors="coerce")
    df = df.sort_values("period")

    fig, (ax_main, ax_ret) = plt.subplots(
        2, 1, figsize=_FIGSIZE_WIDE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    cum = df["cumulative_pnl"] * 100  # convert to %

    # ── Drawdown calculation ──────────────────────────────────────────────
    cum_val = 1 + df["cumulative_pnl"]
    rolling_max = cum_val.cummax()
    drawdown = (cum_val - rolling_max) / rolling_max

    # Shade drawdown periods
    in_dd = drawdown < -0.01
    changes = in_dd.astype(int).diff().fillna(0)
    dd_starts = df.loc[changes == 1, "period"].tolist()
    dd_ends = df.loc[changes == -1, "period"].tolist()
    # Handle edge case where we end in a drawdown
    if len(dd_starts) > len(dd_ends) and len(dd_starts) > 0:
        dd_ends.append(df["period"].iloc[-1])

    for s, e in zip(dd_starts, dd_ends):
        ax_main.axvspan(s, e, alpha=0.12, color=_PALETTE["short"], zorder=0)

    # ── Main PnL line ─────────────────────────────────────────────────────
    ax_main.plot(df["period"], cum, color=_PALETTE["long"],
                 linewidth=2.5, label="Cumulative L-S Return")
    ax_main.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_main.fill_between(df["period"], 0, cum,
                          where=(cum >= 0), alpha=0.15, color=_PALETTE["beat"])
    ax_main.fill_between(df["period"], 0, cum,
                          where=(cum < 0), alpha=0.15, color=_PALETTE["short"])

    ax_main.set_ylabel("Cumulative L-S Return (%)", fontsize=11)
    ax_main.set_title("Long-Short Job Posting Velocity Strategy — Cumulative PnL",
                      fontsize=13, fontweight="bold")

    dd_patch = mpatches.Patch(color=_PALETTE["short"], alpha=0.2, label="Drawdown Period")
    ax_main.legend(handles=[
        plt.Line2D([0], [0], color=_PALETTE["long"], linewidth=2, label="Cumulative L-S"),
        dd_patch
    ], fontsize=9)

    # ── Monthly bar chart ─────────────────────────────────────────────────
    monthly_ret = df["ls_return"] * 100
    pos = monthly_ret >= 0
    ax_ret.bar(df.loc[pos, "period"], monthly_ret[pos],
               width=20, color=_PALETTE["beat"], alpha=0.75)
    ax_ret.bar(df.loc[~pos, "period"], monthly_ret[~pos],
               width=20, color=_PALETTE["short"], alpha=0.75)
    ax_ret.axhline(0, color="black", linewidth=0.8)
    ax_ret.set_ylabel("Monthly L-S Ret (%)", fontsize=9)
    ax_ret.set_xlabel("Month", fontsize=10)

    fig.tight_layout()
    _save(fig, "cumulative_pnl.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Beat rate by signal quintile
# ─────────────────────────────────────────────────────────────────────────────

def plot_beat_rate_by_signal_quintile(aligned_df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of earnings beat rate by signal_strength quintile.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Must contain ``["signal_strength", "beat_flag"]``.

    Returns
    -------
    matplotlib.figure.Figure
        Saved to ``outputs/beat_rate_by_quintile.png``.

    Notes
    -----
    This is the **key validation chart** for the strategy.

    A monotonically increasing bar chart — where companies in the highest
    signal_strength quintile (Q5) have materially higher earnings beat rates
    than those in Q1 — is the clearest visual evidence that the signal has
    predictive content.

    If the bars do not exhibit a clear upward trend, the signal does not
    generalise across the cross-section and the strategy should not be deployed.
    """
    df = aligned_df.dropna(subset=["signal_strength", "beat_flag"]).copy()
    if df.empty:
        logger.warning("No data for quintile beat-rate chart")
        return plt.figure()

    df["quintile"] = pd.qcut(
        df["signal_strength"],
        q=5,
        labels=["Q1\n(Weakest)", "Q2", "Q3", "Q4", "Q5\n(Strongest)"],
        duplicates="drop",
    )

    beat_by_q = (
        df.groupby("quintile", observed=True)["beat_flag"]
        .agg(["mean", "count"])
        .reset_index()
    )
    beat_by_q["beat_rate_pct"] = beat_by_q["mean"] * 100

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = sns.color_palette("Blues", n_colors=len(beat_by_q))
    bars = ax.bar(
        beat_by_q["quintile"].astype(str),
        beat_by_q["beat_rate_pct"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    # Annotate bars with beat rate and n
    for bar, (_, row) in zip(bars, beat_by_q.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['beat_rate_pct']:.1f}%\n(n={int(row['count'])})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Reference line: overall beat rate
    overall_beat = df["beat_flag"].mean() * 100
    ax.axhline(overall_beat, color=_PALETTE["short"], linestyle="--",
               linewidth=1.2, label=f"Overall Beat Rate ({overall_beat:.1f}%)")

    ax.set_ylim(0, min(100, beat_by_q["beat_rate_pct"].max() * 1.25))
    ax.set_xlabel("Signal Strength Quintile", fontsize=11)
    ax.set_ylabel("Earnings Beat Rate (%)", fontsize=11)
    ax.set_title(
        "Earnings Beat Rate by Job Posting Signal Strength Quintile\n"
        "(Monotonically increasing bars → signal has cross-sectional predictive content)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "beat_rate_by_quintile.png")
    return fig
