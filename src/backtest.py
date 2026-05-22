"""
backtest.py
===========
Pandas-based event-study and long-short backtest engine.

No external backtesting libraries are required — everything is computed
in plain pandas/numpy.

Functions
---------
run_event_study           : Compute holding-period returns for each signal event.
run_long_short_backtest   : Monthly L-S portfolio construction and PnL accumulation.
compute_performance_metrics : Sharpe, max drawdown, win rate, sector attribution.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Event study
# ─────────────────────────────────────────────────────────────────────────────

def run_event_study(
    aligned_df: pd.DataFrame,
    price_data: pd.DataFrame,
    holding_days: int = config.DEFAULT_HOLDING_DAYS,
) -> pd.DataFrame:
    """
    Compute buy-and-hold returns for each signal event.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of :func:`src.signal.align_signal_to_earnings`.
        Must contain ``["ticker", "signal_month", "long_signal",
        "short_signal", "signal_strength", "beat_flag", "surprise_pct"]``.
    price_data : pd.DataFrame
        Daily price data with columns ``["ticker", "date", "close"]``.
        ``date`` must be datetime.
    holding_days : int, optional
        Calendar days to hold from signal_month end (default: 60).

    Returns
    -------
    pd.DataFrame
        Columns: ``["ticker", "signal_month", "signal_type", "entry_date",
        "exit_date", "entry_price", "exit_price", "holding_return",
        "beat_flag", "surprise_pct"]``.
        ``holding_return`` is the gross return (e.g. 0.05 = +5%).
        For SHORT events, the return is the *negative* of the raw price return
        (profit from falling price).

    Notes
    -----
    - Only rows where ``long_signal=True`` or ``short_signal=True`` generate events.
    - If price data is unavailable for a ticker on the required dates, the event
      is skipped (not imputed).
    - ``entry_date`` = last trading day of signal_month.
    - ``exit_date`` = entry_date + holding_days, snapped to nearest available date.
    """
    price_data = price_data.copy()
    price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce")
    price_data = price_data.dropna(subset=["date", "close"])
    price_data = price_data.sort_values(["ticker", "date"])

    records: list[dict] = []

    # Process long and short events
    event_df = aligned_df[
        aligned_df["long_signal"] | aligned_df["short_signal"]
    ].copy()

    for _, row in event_df.iterrows():
        ticker = row["ticker"]
        signal_type = "LONG" if row["long_signal"] else "SHORT"

        # Entry: last available trading day of the signal month
        month_end = pd.to_datetime(row["signal_month"], format="%Y%m") + pd.offsets.MonthEnd(0)
        ticker_prices = price_data[price_data["ticker"] == ticker]

        entry_candidates = ticker_prices[ticker_prices["date"] <= month_end]
        if entry_candidates.empty:
            logger.debug("No entry price for %s at %s — skipping", ticker, month_end)
            continue

        entry_row = entry_candidates.iloc[-1]
        entry_date = entry_row["date"]
        entry_price = entry_row["close"]

        # Exit: closest available date >= entry_date + holding_days
        target_exit = entry_date + pd.Timedelta(days=holding_days)
        exit_candidates = ticker_prices[ticker_prices["date"] >= target_exit]
        if exit_candidates.empty:
            logger.debug("No exit price for %s after %s — skipping", ticker, target_exit)
            continue

        exit_row = exit_candidates.iloc[0]
        exit_date = exit_row["date"]
        exit_price = exit_row["close"]

        raw_return = (exit_price - entry_price) / entry_price
        holding_return = raw_return if signal_type == "LONG" else -raw_return

        records.append(
            {
                "ticker": ticker,
                "signal_month": row["signal_month"],
                "signal_type": signal_type,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_return": holding_return,
                "beat_flag": row.get("beat_flag"),
                "surprise_pct": row.get("surprise_pct"),
                "signal_strength": row.get("signal_strength"),
            }
        )

    result = pd.DataFrame(records)
    logger.info("Event study: %d events (%d long, %d short)",
                len(result),
                (result["signal_type"] == "LONG").sum() if not result.empty else 0,
                (result["signal_type"] == "SHORT").sum() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Long-short portfolio backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_long_short_backtest(
    aligned_df: pd.DataFrame,
    jolts_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct an equal-weighted monthly long-short portfolio from signal terciles.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Must contain ``["ticker", "signal_month", "signal_strength",
        "surprise_pct", "long_signal", "short_signal"]``.
    jolts_df : pd.DataFrame, optional
        BLS JOLTS macro data for regime overlay column.

    Returns
    -------
    pd.DataFrame
        Columns: ``["month", "long_return", "short_return", "ls_return",
        "cumulative_pnl", "n_long", "n_short"]``.
        Returns empty DataFrame if fewer than 3 signal months available.

    Notes
    -----
    Portfolio construction:

    - Each month, rank all available companies by ``signal_strength``.
    - **Long book**: top tercile (highest signal_strength).
    - **Short book**: bottom tercile (lowest signal_strength).
    - Weights are equal within each book.
    - L-S return = mean(long_returns) − mean(short_returns).

    The return proxy for each company-month is the forward ``surprise_pct``
    (a proxy for the abnormal return at earnings).  In a full implementation,
    this would be replaced by actual price returns.

    Cumulative PnL is computed as ``(1 + ls_return).cumprod() - 1``.
    """
    df = aligned_df.dropna(subset=["signal_strength", "surprise_pct"]).copy()
    if df["signal_month"].nunique() < 3:
        logger.warning("Fewer than 3 signal months — backtest results unreliable")
        return pd.DataFrame()

    months = sorted(df["signal_month"].unique())
    records: list[dict] = []

    for month in months:
        month_df = df[df["signal_month"] == month].copy()
        if len(month_df) < 3:
            continue

        # Tercile assignment
        n = len(month_df)
        tercile_cutoff = max(1, n // 3)

        ranked = month_df.sort_values("signal_strength", ascending=False)
        long_book = ranked.head(tercile_cutoff)
        short_book = ranked.tail(tercile_cutoff)

        long_ret = long_book["surprise_pct"].mean() / 100  # convert % to decimal
        short_ret = short_book["surprise_pct"].mean() / 100
        ls_ret = long_ret - short_ret

        records.append(
            {
                "month": month,
                "long_return": long_ret,
                "short_return": short_ret,
                "ls_return": ls_ret,
                "n_long": len(long_book),
                "n_short": len(short_book),
            }
        )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["cumulative_pnl"] = (1 + result["ls_return"]).cumprod() - 1

    # Merge JOLTS regime if provided
    if jolts_df is not None and not jolts_df.empty:
        jolts_df = jolts_df.copy()
        jolts_df["month"] = jolts_df["date"].dt.strftime("%Y%m")
        result = result.merge(
            jolts_df[["month", "jolts_openings_rate"]],
            on="month",
            how="left",
        )

    logger.info(
        "L-S backtest: %d months, final cumPnL=%.2f%%",
        len(result),
        result["cumulative_pnl"].iloc[-1] * 100 if len(result) > 0 else 0,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    backtest_results_df: pd.DataFrame,
    aligned_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute strategy-level performance statistics.

    Parameters
    ----------
    backtest_results_df : pd.DataFrame
        Output of :func:`run_long_short_backtest`.
        Must contain ``["ls_return", "cumulative_pnl"]``.
    aligned_df : pd.DataFrame, optional
        Signal-earnings aligned data for sector-level attribution.

    Returns
    -------
    dict
        Keys:

        - ``"annualized_return"`` — geometric annualised L-S return
        - ``"annualized_sharpe"`` — Sharpe ratio (252-day basis, zero risk-free)
        - ``"max_drawdown"`` — maximum peak-to-trough drawdown
        - ``"max_dd_peak_month"`` — month of drawdown peak
        - ``"max_dd_trough_month"`` — month of drawdown trough
        - ``"win_rate"`` — fraction of months with positive L-S return
        - ``"avg_ls_return"`` — average monthly L-S return
        - ``"total_periods"`` — number of months
        - ``"jolts_correlation"`` — correlation of L-S return with JOLTS rate
          (if available)
        - ``"sector_hit_rates"`` — dict of beat rates by sector (if available)

    Notes
    -----
    Annualised Sharpe uses 12 periods (months) per year.  The strategy is
    evaluated on pre-cost gross returns; transaction costs and market-impact
    are not modelled.
    """
    df = backtest_results_df.dropna(subset=["ls_return"]).copy()

    if df.empty:
        logger.error("No data for performance metrics computation")
        return {}

    n = len(df)
    returns = df["ls_return"]

    # Annualised return (geometric, monthly periods)
    final_pnl = df["cumulative_pnl"].iloc[-1]
    ann_return = (1 + final_pnl) ** (12 / n) - 1

    # Sharpe (monthly frequency, 12 periods/year)
    avg_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    sharpe = (avg_ret / std_ret * np.sqrt(12)) if std_ret > 0 else float("nan")

    # Maximum drawdown
    cum_val = 1 + df["cumulative_pnl"]
    rolling_max = cum_val.cummax()
    drawdown = (cum_val - rolling_max) / rolling_max
    max_dd = drawdown.min()
    peak_idx = drawdown.idxmin()
    peak_month = df.loc[peak_idx, "month"] if peak_idx in df.index else "n/a"
    trough_month = df.loc[peak_idx, "month"] if peak_idx in df.index else "n/a"
    # Refine peak
    if peak_idx in df.index:
        peak_loc = cum_val.loc[:peak_idx].idxmax()
        peak_month = df.loc[peak_loc, "month"] if peak_loc in df.index else "n/a"
        trough_month = df.loc[peak_idx, "month"]

    win_rate = (returns > 0).mean()

    # JOLTS correlation
    jolts_corr = float("nan")
    if "jolts_openings_rate" in df.columns and df["jolts_openings_rate"].notna().any():
        jolts_corr = float(df["ls_return"].corr(df["jolts_openings_rate"]))

    # Sector hit rates
    sector_hit_rates: dict[str, float] = {}
    if aligned_df is not None and "sector" in aligned_df.columns:
        gb = aligned_df.groupby("sector")["beat_flag"]
        sector_hit_rates = {
            str(sector): float(grp.mean())
            for sector, grp in gb
            if grp.notna().any()
        }

    metrics = {
        "annualized_return": ann_return,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_peak_month": peak_month,
        "max_dd_trough_month": trough_month,
        "win_rate": win_rate,
        "avg_ls_return": avg_ret,
        "total_periods": n,
        "jolts_correlation": jolts_corr,
        "sector_hit_rates": sector_hit_rates,
    }

    # ── Pretty print ─────────────────────────────────────────────────────────
    sep = "═" * 52
    print(f"\n{sep}")
    print("  STRATEGY PERFORMANCE SUMMARY")
    print(sep)
    print(f"  Periods (months)     : {n}")
    print(f"  Annualised Return    : {ann_return * 100:>8.2f}%")
    print(f"  Annualised Sharpe    : {sharpe:>8.3f}")
    print(f"  Max Drawdown         : {max_dd * 100:>8.2f}%")
    print(f"    Peak month         : {peak_month}")
    print(f"    Trough month       : {trough_month}")
    print(f"  Win Rate             : {win_rate * 100:>8.2f}%")
    print(f"  Avg Monthly L-S Ret  : {avg_ret * 100:>8.3f}%")
    if not np.isnan(jolts_corr):
        print(f"  JOLTS Correlation    : {jolts_corr:>8.3f}")
    if sector_hit_rates:
        print("\n  Beat Rate by Sector:")
        for s, r in sorted(sector_hit_rates.items()):
            print(f"    {s:<20}: {r * 100:.1f}%")
    print(sep)

    logger.info(
        "Sharpe=%.3f, MaxDD=%.2f%%, WinRate=%.2f%%",
        sharpe,
        max_dd * 100,
        win_rate * 100,
    )
    return metrics
