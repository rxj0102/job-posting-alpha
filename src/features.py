"""
features.py
===========
Feature engineering for the Job Posting Velocity strategy.

Pipeline
--------
1. Classify raw job title text into functional categories (ENGINEERING, PRODUCT, …).
2. Compute per-company functional decomposition and the BULLISH signal ratio.
3. Compute month-over-month net-new postings and pct change.
4. Normalise by company headcount.
5. Compute the rolling z-score of hiring acceleration per company.
6. Build the composite long/short signal.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Category ordering for DataFrame columns
_CATEGORIES = ["ENGINEERING", "PRODUCT", "SALES", "GA", "LEGAL", "HR", "OPERATIONS", "OTHER"]
_BULLISH_CATS = {"ENGINEERING", "PRODUCT", "SALES"}
_BEARISH_CATS = {"LEGAL", "GA", "HR"}


# ─────────────────────────────────────────────────────────────────────────────
# Job title classification
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_taxonomy() -> pd.DataFrame:
    """Load job function taxonomy CSV once and cache it in memory."""
    path = config.JOB_TAXONOMY_CSV
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {path}")
    df = pd.read_csv(path)
    df["keyword"] = df["keyword"].str.lower().str.strip()
    return df


def classify_job_titles(titles_list: list[str]) -> dict[str, int]:
    """
    Classify a list of raw job title strings into functional categories.

    Parameters
    ----------
    titles_list : list[str]
        Job title strings extracted from a single month's snapshot.

    Returns
    -------
    dict[str, int]
        Mapping of ``function_category → count``.
        Each title contributes exactly 1 count to the *first matching* category
        found (keyword list is ordered most-specific to least-specific in the CSV).
        Unmatched titles are counted under ``"OTHER"``.

    Notes
    -----
    - Matching is case-insensitive substring search.
    - The taxonomy CSV is loaded once and LRU-cached.
    """
    taxonomy = _load_taxonomy()
    counts: dict[str, int] = {cat: 0 for cat in _CATEGORIES}

    for title in titles_list:
        title_lc = title.lower()
        matched = False
        for _, row in taxonomy.iterrows():
            if row["keyword"] in title_lc:
                cat = row["function_category"].upper()
                if cat in counts:
                    counts[cat] += 1
                else:
                    counts["OTHER"] += 1
                matched = True
                break
        if not matched:
            counts["OTHER"] += 1

    return counts


def compute_functional_decomposition(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply :func:`classify_job_titles` to every row of the timeseries DataFrame.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        Must contain columns ``["ticker", "year_month", "job_titles_raw"]``.
        ``job_titles_raw`` is expected as a JSON-encoded list of strings.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with columns:

        - ``engineering_count``, ``product_count``, ``sales_count``,
          ``ga_count``, ``legal_count``, ``hr_count``, ``operations_count``,
          ``other_count``
        - ``total_count`` — sum of all category counts
        - ``signal_ratio`` — (engineering + product + sales) / total_count;
          values in [0, 1]; NaN if total_count == 0

    Notes
    -----
    Rows where ``job_titles_raw`` is missing, null, or empty list produce all-zero
    category counts and ``signal_ratio = NaN``.
    """
    df = timeseries_df.copy()

    cat_records: list[dict] = []
    for _, row in df.iterrows():
        raw = row.get("job_titles_raw", "[]")
        try:
            titles: list[str] = json.loads(raw) if pd.notna(raw) else []
        except (json.JSONDecodeError, TypeError):
            titles = []

        counts = classify_job_titles(titles)
        cat_records.append(counts)

    decomp_df = pd.DataFrame(cat_records)
    decomp_df.columns = [f"{c.lower()}_count" for c in decomp_df.columns]

    # Merge back
    df = pd.concat([df.reset_index(drop=True), decomp_df], axis=1)

    df["total_count"] = decomp_df.sum(axis=1)
    bullish_sum = (
        df["engineering_count"] + df["product_count"] + df["sales_count"]
    )
    df["signal_ratio"] = np.where(
        df["total_count"] > 0,
        bullish_sum / df["total_count"],
        np.nan,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Posting velocity features
# ─────────────────────────────────────────────────────────────────────────────

def compute_net_new_postings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-over-month change in posting count per ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``["ticker", "year_month", "posting_count"]``.
        ``year_month`` should be sortable (e.g. ``"202301"``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        - ``net_new`` — absolute month-over-month change in posting_count
        - ``pct_change`` — percentage month-over-month change (NaN for first row
          per ticker or when prior month count is 0)

    Notes
    -----
    Computation is done per-ticker group to avoid cross-company contamination.
    Rows without a preceding month (first observation per ticker) receive NaN.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "year_month"])

    df["net_new"] = df.groupby("ticker")["posting_count"].diff()
    prior = df.groupby("ticker")["posting_count"].shift(1)
    df["pct_change"] = np.where(
        prior.notna() & (prior != 0),
        df["net_new"] / prior * 100,
        np.nan,
    )
    return df


def normalize_by_headcount(
    df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise posting counts by company headcount.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame containing ``["ticker", "posting_count", "net_new"]``.
    universe_df : pd.DataFrame
        Company universe with ``["ticker", "avg_headcount"]``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        - ``avg_headcount`` — joined from universe_df
        - ``normalized_posting_rate`` — posting_count / avg_headcount
        - ``normalized_net_new`` — net_new / avg_headcount

    Notes
    -----
    Normalising by headcount is critical for cross-sectional comparisons.
    A 100-opening increase at a 5,000-person company is very different from
    the same increase at a 500,000-person company.
    """
    df = df.copy()
    hc = universe_df[["ticker", "avg_headcount"]].drop_duplicates("ticker")
    df = df.merge(hc, on="ticker", how="left")

    df["normalized_posting_rate"] = np.where(
        df["avg_headcount"].notna() & (df["avg_headcount"] > 0),
        df["posting_count"] / df["avg_headcount"],
        np.nan,
    )
    df["normalized_net_new"] = np.where(
        df["avg_headcount"].notna() & (df["avg_headcount"] > 0),
        df["net_new"] / df["avg_headcount"],
        np.nan,
    )
    return df


def compute_hiring_acceleration_zscore(
    df: pd.DataFrame,
    window: int = config.ZSCORE_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Compute a per-ticker rolling z-score of headcount-normalised net-new postings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``["ticker", "year_month", "normalized_net_new"]``.
    window : int, optional
        Rolling window in months (default: 12).

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        - ``rolling_mean`` — rolling mean of normalized_net_new
        - ``rolling_std`` — rolling std of normalized_net_new
        - ``hiring_accel_zscore`` — (current - rolling_mean) / rolling_std
        - ``high_acceleration`` — boolean; True if hiring_accel_zscore
          exceeds ``config.HIRING_ACCEL_ZSCORE_THRESHOLD``

    Notes
    -----
    Requires at least ``config.ZSCORE_MIN_PERIODS`` observations within the
    window before producing a non-NaN z-score (avoids spurious early signals).

    The rolling window is computed *per ticker* over sorted ``year_month`` values.
    """
    df = df.copy().sort_values(["ticker", "year_month"])

    min_periods = config.ZSCORE_MIN_PERIODS

    def _zscore_group(grp: pd.DataFrame) -> pd.DataFrame:
        series = grp["normalized_net_new"]
        roll = series.rolling(window=window, min_periods=min_periods)
        grp = grp.copy()
        grp["rolling_mean"] = roll.mean()
        grp["rolling_std"] = roll.std()
        grp["hiring_accel_zscore"] = np.where(
            grp["rolling_std"].notna() & (grp["rolling_std"] > 0),
            (series - grp["rolling_mean"]) / grp["rolling_std"],
            np.nan,
        )
        return grp

    df = df.groupby("ticker", group_keys=False).apply(_zscore_group)
    df["high_acceleration"] = (
        df["hiring_accel_zscore"] > config.HIRING_ACCEL_ZSCORE_THRESHOLD
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Composite signal
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the composite long/short signal from acceleration z-score and
    functional decomposition ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``["hiring_accel_zscore", "high_acceleration", "signal_ratio"]``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        - ``long_signal`` — True if *both* high_acceleration AND
          signal_ratio > ``config.SIGNAL_RATIO_THRESHOLD``.
          Captures firms that are accelerating hiring AND deploying capital
          into revenue-generating (bullish) functions.
        - ``short_signal`` — True if the firm is *decelerating* sharply
          AND the remaining mix skews toward overhead/legal functions.
        - ``signal_strength`` — continuous signal: hiring_accel_zscore ×
          signal_ratio.  Use for portfolio ranking (top / bottom tercile).

    Notes
    -----
    The composite signal intentionally requires *two* conditions for a long:

    1. **Velocity**: hiring is accelerating faster than the company's own
       trailing 12-month history (z-score threshold).
    2. **Quality**: the new hiring is predominantly in growth-oriented
       functions (ENGINEERING / PRODUCT / SALES) rather than overhead.

    This dual filter reduces false positives from, e.g., a company
    rapidly hiring compliance officers ahead of a regulatory settlement.
    """
    df = df.copy()

    # Long signal: acceleration + growth-function tilt
    df["long_signal"] = (
        df["high_acceleration"].fillna(False)
        & (df["signal_ratio"] > config.SIGNAL_RATIO_THRESHOLD)
    )

    # Short signal: deceleration + overhead/legal tilt
    df["short_signal"] = (
        (df["hiring_accel_zscore"] < -config.HIRING_ACCEL_ZSCORE_THRESHOLD)
        & (df["signal_ratio"] < 0.35)
    ).fillna(False)

    # Continuous ranking signal
    df["signal_strength"] = df["hiring_accel_zscore"] * df["signal_ratio"]

    n_long = df["long_signal"].sum()
    n_short = df["short_signal"].sum()
    logger.info(
        "Composite signal: %d long events, %d short events across %d company-months",
        n_long,
        n_short,
        len(df),
    )
    return df
