"""
signal.py
=========
Aligns the job-posting signal to forward earnings outcomes and tests
whether the signal has statistically significant predictive content.

Key functions
-------------
align_signal_to_earnings   : Join signal months to forward earnings quarters.
run_ttest_validation        : t-test and proportion test (primary hypothesis test).
run_ols_regression          : OLS regression of surprise_pct on signal features.
compute_information_coefficient : Spearman IC between signal_strength and surprise_pct.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal–earnings alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_signal_to_earnings(
    signal_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    lead_quarters: int = 1,
) -> pd.DataFrame:
    """
    Match each signal month to the earnings release approximately
    ``lead_quarters`` quarters in the future.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Must contain columns ``["ticker", "year_month", "long_signal",
        "short_signal", "signal_strength", "hiring_accel_zscore",
        "signal_ratio"]``.
        ``year_month`` in ``"YYYYMM"`` format (string).
    earnings_df : pd.DataFrame
        Must contain columns ``["ticker", "earnings_date", "beat_flag",
        "surprise_pct"]``.
        ``earnings_date`` as datetime (timezone-naive or UTC).
    lead_quarters : int, optional
        Number of quarters ahead to look for the forward earnings event
        (default: 1, i.e. ~90 days).

    Returns
    -------
    pd.DataFrame
        Columns: ``["ticker", "signal_month", "earnings_date", "beat_flag",
        "surprise_pct", "long_signal", "short_signal", "signal_strength",
        "hiring_accel_zscore", "signal_ratio"]``.
        Only rows with a matching forward earnings event are retained.

    Notes
    -----
    The matching logic looks for the *first* earnings date that falls in the
    window ``[signal_month_end + 30, signal_month_end + lead_quarters * 120]``.
    A ±30-day buffer on each side handles quarterly reporting variability.
    """
    target_days = lead_quarters * 90

    # Normalise earnings_date to tz-naive
    earnings_df = earnings_df.copy()
    earnings_df["earnings_date"] = pd.to_datetime(
        earnings_df["earnings_date"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    earnings_df = earnings_df.dropna(subset=["earnings_date"])

    # Convert year_month → Timestamp (last day of month)
    signal_df = signal_df.copy()
    signal_df["signal_date"] = pd.to_datetime(
        signal_df["year_month"], format="%Y%m", errors="coerce"
    ) + pd.offsets.MonthEnd(0)
    signal_df = signal_df.dropna(subset=["signal_date"])

    records: list[dict] = []
    for _, sig_row in signal_df.iterrows():
        ticker = sig_row["ticker"]
        sig_date = sig_row["signal_date"]

        # Window: 30 days after signal → target + 120 days buffer
        win_start = sig_date + pd.Timedelta(days=30)
        win_end = sig_date + pd.Timedelta(days=target_days + 120)

        ticker_earnings = earnings_df[earnings_df["ticker"] == ticker]
        candidates = ticker_earnings[
            (ticker_earnings["earnings_date"] >= win_start)
            & (ticker_earnings["earnings_date"] <= win_end)
        ].sort_values("earnings_date")

        if candidates.empty:
            continue

        nearest = candidates.iloc[0]
        records.append(
            {
                "ticker": ticker,
                "signal_month": sig_row["year_month"],
                "earnings_date": nearest["earnings_date"],
                "beat_flag": nearest["beat_flag"],
                "surprise_pct": nearest["surprise_pct"],
                "long_signal": sig_row.get("long_signal", False),
                "short_signal": sig_row.get("short_signal", False),
                "signal_strength": sig_row.get("signal_strength", np.nan),
                "hiring_accel_zscore": sig_row.get("hiring_accel_zscore", np.nan),
                "signal_ratio": sig_row.get("signal_ratio", np.nan),
            }
        )

    aligned = pd.DataFrame(records)
    logger.info(
        "Aligned %d signal months to forward earnings (lead=%dQ)",
        len(aligned),
        lead_quarters,
    )
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis tests
# ─────────────────────────────────────────────────────────────────────────────

def run_ttest_validation(aligned_df: pd.DataFrame) -> dict:
    """
    Test whether the long-signal group has a higher earnings surprise than
    the non-signal group.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of :func:`align_signal_to_earnings`.
        Must contain ``["long_signal", "surprise_pct", "beat_flag"]``.

    Returns
    -------
    dict
        Keys: ``"n_signal"``, ``"n_control"``, ``"mean_surprise_signal"``,
        ``"mean_surprise_control"``, ``"beat_rate_signal"``,
        ``"beat_rate_control"``, ``"tstat"``, ``"pvalue"``,
        ``"prop_zstat"``, ``"prop_pvalue"``.

    Notes
    -----
    **Primary Hypothesis Test**

    - H₀: Mean earnings surprise in signal group = mean surprise in control group
    - H₁: Mean earnings surprise in signal group > mean surprise in control group

    If ``p < 0.05`` on a one-tailed Welch t-test, the signal has statistically
    significant predictive content at the 5% level.

    If the sample is too small (< 15 observations per group) or the p-value
    does not pass the threshold, treat the result as inconclusive and do
    not proceed to backtest construction — the signal does not replicate.

    Also runs ``scipy.stats.proportions_ztest`` on the binary beat_flag to
    test whether the long-signal group has a higher hit rate.
    """
    df = aligned_df.dropna(subset=["surprise_pct", "beat_flag"])
    signal_grp = df[df["long_signal"] == True]["surprise_pct"]
    control_grp = df[df["long_signal"] == False]["surprise_pct"]

    n_sig = len(signal_grp)
    n_ctl = len(control_grp)

    mean_sig = signal_grp.mean() if n_sig > 0 else float("nan")
    mean_ctl = control_grp.mean() if n_ctl > 0 else float("nan")

    beat_sig = df[df["long_signal"] == True]["beat_flag"].mean() if n_sig > 0 else float("nan")
    beat_ctl = df[df["long_signal"] == False]["beat_flag"].mean() if n_ctl > 0 else float("nan")

    # Welch t-test (does not assume equal variances)
    if n_sig >= 2 and n_ctl >= 2:
        tstat, pvalue = stats.ttest_ind(signal_grp, control_grp, equal_var=False)
    else:
        tstat, pvalue = float("nan"), float("nan")

    # Proportions z-test for beat rates
    prop_zstat = float("nan")
    prop_pvalue = float("nan")
    if n_sig >= 5 and n_ctl >= 5:
        try:
            from statsmodels.stats.proportion import proportions_ztest
            count = np.array([
                df[df["long_signal"] == True]["beat_flag"].sum(),
                df[df["long_signal"] == False]["beat_flag"].sum(),
            ])
            nobs = np.array([n_sig, n_ctl])
            prop_zstat, prop_pvalue = proportions_ztest(count, nobs, alternative="larger")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Proportions z-test failed: %s", exc)

    results = {
        "n_signal": n_sig,
        "n_control": n_ctl,
        "mean_surprise_signal": mean_sig,
        "mean_surprise_control": mean_ctl,
        "beat_rate_signal": beat_sig,
        "beat_rate_control": beat_ctl,
        "tstat": tstat,
        "pvalue": pvalue,
        "prop_zstat": prop_zstat,
        "prop_pvalue": prop_pvalue,
    }

    # ── Pretty print ─────────────────────────────────────────────────────────
    header = f"{'Group':<14} {'N':>6} {'Mean Surprise %':>16} {'Beat Rate':>10} {'t-stat':>9} {'p-value':>9}"
    sep = "─" * len(header)
    row_sig = (
        f"{'LONG SIGNAL':<14} {n_sig:>6} {mean_sig:>16.2f} "
        f"{beat_sig:>10.3f} {tstat:>9.3f} {pvalue:>9.4f}"
    )
    row_ctl = (
        f"{'CONTROL':<14} {n_ctl:>6} {mean_ctl:>16.2f} "
        f"{beat_ctl:>10.3f} {'':>9} {'':>9}"
    )
    print(f"\n{sep}\n{header}\n{sep}\n{row_sig}\n{row_ctl}\n{sep}")
    sig_str = "✓ SIGNIFICANT" if (not np.isnan(pvalue) and pvalue < 0.05) else "✗ not significant"
    print(f"t-test (one-tailed): t={tstat:.3f}, p={pvalue:.4f} → {sig_str}")
    print(f"Proportions z-test: z={prop_zstat:.3f}, p={prop_pvalue:.4f}")
    print(sep)

    logger.info("t-test results: t=%.3f, p=%.4f", tstat, pvalue)
    return results


def run_ols_regression(aligned_df: pd.DataFrame, jolts_df: Optional[pd.DataFrame] = None) -> object:
    """
    Run OLS regression of earnings surprise on signal features.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of :func:`align_signal_to_earnings`.
        Must contain ``["surprise_pct", "hiring_accel_zscore", "signal_ratio"]``.
    jolts_df : pd.DataFrame, optional
        BLS JOLTS macro data with ``["date", "jolts_openings_rate"]``.
        If provided, the macro overlay is merged and included as a regressor.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.  Full summary is printed to stdout.

    Notes
    -----
    Model specification::

        surprise_pct ~ hiring_accel_zscore + signal_ratio
                     + [jolts_openings_rate]   (if available)
                     + sector_dummy            (if "sector" column present)

    Sector dummies are included to control for industry-level earnings
    cyclicality (e.g. technology firms tend to beat estimates more often
    than utilities).
    """
    import statsmodels.api as sm

    df = aligned_df.dropna(subset=["surprise_pct", "hiring_accel_zscore", "signal_ratio"])

    y = df["surprise_pct"]
    X_cols: list[str] = ["hiring_accel_zscore", "signal_ratio"]

    # Macro overlay
    if jolts_df is not None and not jolts_df.empty:
        jolts_df = jolts_df.copy()
        jolts_df["year_month"] = jolts_df["date"].dt.strftime("%Y%m")
        df = df.merge(
            jolts_df[["year_month", "jolts_openings_rate"]],
            on="year_month",
            how="left",
        )
        if "jolts_openings_rate" in df.columns and df["jolts_openings_rate"].notna().any():
            X_cols.append("jolts_openings_rate")

    # Sector dummies
    if "sector" in df.columns:
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=True)
        df = pd.concat([df, sector_dummies], axis=1)
        X_cols.extend([c for c in sector_dummies.columns])

    df = df.dropna(subset=X_cols)
    y = df["surprise_pct"]
    X = sm.add_constant(df[X_cols])

    model = sm.OLS(y, X).fit(cov_type="HC3")  # Heteroskedasticity-robust SEs
    print("\n" + "=" * 60)
    print("OLS Regression: surprise_pct ~ signal features")
    print("=" * 60)
    print(model.summary())
    logger.info("OLS R²=%.4f, N=%d", model.rsquared, int(model.nobs))
    return model


def compute_information_coefficient(aligned_df: pd.DataFrame) -> float:
    """
    Compute the Spearman rank IC between signal_strength and forward surprise_pct.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Must contain ``["signal_strength", "surprise_pct"]``.

    Returns
    -------
    float
        Spearman IC value (correlation coefficient ρ in [−1, 1]).

    Notes
    -----
    **Interpretation Benchmark**

    An IC above 0.05 is considered economically meaningful for an
    alternative-data signal in the academic literature (Hou, Xue, Zhang 2020).
    Most institutional-grade alternative-data signals operate with ICs in the
    range 0.03–0.15.  An IC below 0.02 should not be used in production.

    Spearman IC is preferred over Pearson because it is robust to outliers in
    the earnings surprise distribution (large misses/beats can have high leverage
    in Pearson correlation).
    """
    df = aligned_df.dropna(subset=["signal_strength", "surprise_pct"])
    if len(df) < 5:
        logger.warning("Too few observations (%d) for IC computation", len(df))
        return float("nan")

    ic, p_value = stats.spearmanr(df["signal_strength"], df["surprise_pct"])
    print(f"\nInformation Coefficient (Spearman ρ): IC = {ic:.4f}, p = {p_value:.4f}")
    if abs(ic) >= 0.05:
        print("  → IC ≥ 0.05: signal has meaningful cross-sectional predictive content.")
    else:
        print("  → IC < 0.05: signal predictive content is weak or absent.")

    logger.info("IC = %.4f, p = %.4f (n=%d)", ic, p_value, len(df))
    return float(ic)
