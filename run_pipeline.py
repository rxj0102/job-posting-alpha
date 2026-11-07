"""
run_pipeline.py
===============
End-to-end demonstration pipeline for the Job Posting Velocity strategy.

Runs entirely on *synthetic* data so no network access is required.
All outputs (charts + CSVs) are written to outputs/ and data/processed/.

Usage
-----
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --no-charts  # skip chart generation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import config

logging.basicConfig(
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    level=logging.INFO,
)
logger = logging.getLogger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_timeseries(universe: pd.DataFrame, n_months: int = 30) -> pd.DataFrame:
    """
    Generate synthetic monthly job-posting timeseries for the full universe.

    Each company receives a realistic base count (proportional to headcount),
    a stochastic trend, and company-specific hiring cycles.  Three companies
    (AMZN, MSFT, NVDA) are seeded to produce strong long signals so that the
    validation charts show a clear monotonic relationship.
    """
    rng = np.random.default_rng(2024)
    months = pd.date_range("2022-01", periods=n_months, freq="MS")
    month_strs = months.strftime("%Y%m").tolist()

    title_pools = {
        "ENGINEERING": [
            "senior software engineer", "backend engineer", "data scientist",
            "ml engineer", "platform engineer", "infrastructure engineer",
            "research scientist", "applied scientist", "devops engineer",
        ],
        "PRODUCT": [
            "product manager", "senior product manager", "ux designer",
            "product designer", "product analyst", "ux researcher",
        ],
        "SALES": [
            "account executive", "enterprise account manager",
            "business development", "customer success manager", "sales engineer",
        ],
        "LEGAL": [
            "corporate counsel", "compliance officer", "general counsel",
            "privacy counsel", "legal operations",
        ],
        "HR": [
            "hr business partner", "talent acquisition", "recruiter",
            "compensation analyst", "hr director",
        ],
        "GA": [
            "financial analyst", "controller", "administrative assistant",
            "facilities manager", "accountant",
        ],
        "OPERATIONS": [
            "program manager", "project manager", "supply chain analyst",
            "business analyst", "operations manager",
        ],
    }

    # Companies with strong engineerings-tilted hiring acceleration signal
    bullish_tickers = {"AMZN", "MSFT", "NVDA", "GOOGL", "META"}

    frames = []
    for i, row in enumerate(universe.itertuples()):
        ticker = row.ticker
        seed = i * 37
        rng_t = np.random.default_rng(seed)

        base = max(80, int(row.avg_headcount / 500))
        trend = np.cumsum(rng_t.normal(0, 3, n_months)).astype(int)
        cycle = (np.sin(np.linspace(0, 4 * np.pi, n_months)) * base * 0.12).astype(int)

        if ticker in bullish_tickers:
            # Build in an acceleration in the last 12 months
            accel = np.zeros(n_months)
            accel[n_months // 2 :] = np.linspace(0, base * 0.35, n_months // 2)
            counts = np.clip(base + trend + cycle + accel.astype(int), 30, 5000)
            bullish_share = rng_t.uniform(0.60, 0.80)  # predominantly ENG/PRODUCT/SALES
        else:
            counts = np.clip(base + trend + cycle, 30, 5000)
            bullish_share = rng_t.uniform(0.30, 0.55)

        for ym, cnt in zip(month_strs, counts):
            cnt = int(cnt)
            # Draw titles proportionally from categories
            n_samples = min(cnt, 60)
            titles = []
            cat_shares = _category_shares(bullish_share, rng_t)
            for cat, pool in title_pools.items():
                n_cat = int(n_samples * cat_shares.get(cat, 0))
                titles.extend(rng_t.choice(pool, size=n_cat, replace=True).tolist())
            rng_t.shuffle(titles)

            frames.append(
                {
                    "ticker": ticker,
                    "year_month": ym,
                    "posting_count": float(cnt),
                    "job_titles_raw": json.dumps(titles[:n_samples]),
                    "extraction_confidence": float(rng_t.choice([0.5, 1.0], p=[0.2, 0.8])),
                }
            )

    return pd.DataFrame(frames)


def _category_shares(bullish_share: float, rng: np.random.Generator) -> dict:
    """Split the total share into per-category fractions."""
    # Bullish cats: ENGINEERING, PRODUCT, SALES
    eng = bullish_share * rng.dirichlet([3, 1.5, 1.5])[0]
    prod = bullish_share * rng.dirichlet([3, 1.5, 1.5])[1]
    sales = bullish_share - eng - prod
    bearish = 1.0 - bullish_share
    legal = bearish * 0.2
    hr = bearish * 0.25
    ga = bearish * 0.2
    ops = bearish * 0.35
    return {
        "ENGINEERING": max(0, eng),
        "PRODUCT": max(0, prod),
        "SALES": max(0, sales),
        "LEGAL": max(0, legal),
        "HR": max(0, hr),
        "GA": max(0, ga),
        "OPERATIONS": max(0, ops),
    }


def _make_earnings(universe: pd.DataFrame, n_quarters: int = 16) -> pd.DataFrame:
    """
    Generate synthetic quarterly EPS history.

    Bullish-signal companies have a higher mean surprise in the last 6 quarters
    to create a real (but modest) signal effect.
    """
    rng = np.random.default_rng(42)
    bullish_tickers = {"AMZN", "MSFT", "NVDA", "GOOGL", "META"}
    frames = []
    for row in universe.itertuples():
        ticker = row.ticker
        dates = pd.date_range("2019-01-15", periods=n_quarters, freq="QS")
        eps_est = rng.uniform(1.0, 8.0, n_quarters)
        # Bullish group: higher mean beat, especially in last 6Q
        if ticker in bullish_tickers:
            surprise_early = rng.normal(0.01, 0.06, n_quarters - 6)
            surprise_late = rng.normal(0.055, 0.05, 6)   # stronger beats recently
            surprise = np.concatenate([surprise_early, surprise_late])
        else:
            surprise = rng.normal(0.005, 0.07, n_quarters)

        eps_act = eps_est * (1 + surprise)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "earnings_date": dates,
                    "eps_estimate": eps_est,
                    "eps_actual": eps_act,
                    "surprise_pct": surprise * 100,
                    "beat_flag": eps_act > eps_est,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def run(generate_charts: bool = True) -> None:
    from src import features, signal, backtest, visualize

    sep = "═" * 64

    # ── 1. Load universe ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 1 — Load company universe")
    print(sep)
    universe = pd.read_csv(config.COMPANY_UNIVERSE_CSV)
    print(f"  {len(universe)} companies across {universe['sector'].nunique()} sectors")

    # ── 2. Generate synthetic timeseries ─────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 2 — Synthetic job-posting timeseries (30 months)")
    print(sep)
    raw_ts = _make_timeseries(universe, n_months=30)
    raw_ts.to_csv(config.UNIVERSE_TIMESERIES_CSV, index=False)
    print(f"  {len(raw_ts)} company-month rows saved → {config.UNIVERSE_TIMESERIES_CSV}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 3 — Feature engineering")
    print(sep)
    df = features.compute_functional_decomposition(raw_ts)
    df = features.compute_net_new_postings(df)
    df = features.normalize_by_headcount(df, universe)
    df = features.compute_hiring_acceleration_zscore(df)
    df = features.build_composite_signal(df)

    enriched_path = config.PROCESSED_DIR / "enriched_features.csv"
    df.to_csv(enriched_path, index=False)
    print(f"  Feature matrix: {df.shape}  →  {enriched_path}")
    print(f"  Long signals : {int(df['long_signal'].sum())} company-months")
    print(f"  Short signals: {int(df['short_signal'].sum())} company-months")

    # ── 4. Synthetic earnings ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 4 — Synthetic earnings history")
    print(sep)
    earnings_df = _make_earnings(universe, n_quarters=20)
    earnings_df["earnings_date"] = pd.to_datetime(earnings_df["earnings_date"])
    earnings_path = config.PROCESSED_DIR / "synthetic_earnings.csv"
    earnings_df.to_csv(earnings_path, index=False)
    print(f"  {len(earnings_df)} earnings observations  →  {earnings_path}")

    # ── 5. Signal alignment ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 5 — Signal → forward-earnings alignment (1Q lead)")
    print(sep)
    aligned = signal.align_signal_to_earnings(df, earnings_df, lead_quarters=1)
    aligned = aligned.merge(universe[["ticker", "sector"]], on="ticker", how="left")
    aligned_path = config.PROCESSED_DIR / "aligned_signal_earnings.csv"
    aligned.to_csv(aligned_path, index=False)
    print(f"  {len(aligned)} aligned signal-earnings pairs  →  {aligned_path}")

    # ── 6. Hypothesis testing ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 6 — Hypothesis tests")
    print(sep)
    ttest_results = signal.run_ttest_validation(aligned)
    ic_val = signal.compute_information_coefficient(aligned)

    try:
        ols_model = signal.run_ols_regression(aligned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OLS skipped (insufficient data): %s", exc)
        ols_model = None

    # ── 7. Backtest ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STAGE 7 — Long-short backtest")
    print(sep)
    bt = backtest.run_long_short_backtest(aligned)
    if bt.empty:
        logger.warning("Backtest returned empty — insufficient signal months")
    else:
        metrics = backtest.compute_performance_metrics(bt, aligned)
        bt_path = config.PROCESSED_DIR / "backtest_results.csv"
        bt.to_csv(bt_path, index=False)
        print(f"  Backtest results  →  {bt_path}")

    # ── 8. Charts ─────────────────────────────────────────────────────────────
    if generate_charts:
        print(f"\n{sep}")
        print("  STAGE 8 — Generating charts  →  outputs/")
        print(sep)

        # Chart 1 — posting timeseries (MSFT + NVDA)
        for tkr in ["MSFT", "NVDA"]:
            fig = visualize.plot_posting_timeseries(df, tkr)
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  ✓  posting_timeseries_{tkr}.png")

        # Chart 2 — functional decomposition heatmap
        fig = visualize.plot_functional_decomposition_heatmap(df)
        plt.close(fig)
        print("  ✓  functional_decomposition_heatmap.png")

        # Chart 3 — z-score distribution
        fig = visualize.plot_zscore_distribution(df)
        plt.close(fig)
        print("  ✓  zscore_distribution.png")

        # Chart 4 — signal vs earnings surprise scatter
        fig = visualize.plot_signal_vs_earnings_surprise(aligned, ic_value=ic_val)
        plt.close(fig)
        print("  ✓  signal_vs_surprise_scatter.png")

        # Chart 5 — cumulative PnL
        if not bt.empty:
            fig = visualize.plot_cumulative_pnl(bt)
            plt.close(fig)
            print("  ✓  cumulative_pnl.png")

        # Chart 6 — beat rate by quintile
        fig = visualize.plot_beat_rate_by_signal_quintile(aligned)
        plt.close(fig)
        print("  ✓  beat_rate_by_quintile.png")

    print(f"\n{sep}")
    print("  PIPELINE COMPLETE")
    print(f"  Outputs → {config.OUTPUTS_DIR}")
    print(f"  Processed data → {config.PROCESSED_DIR}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Job Posting Velocity pipeline on synthetic data."
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (faster, useful for CI checks).",
    )
    args = parser.parse_args()
    run(generate_charts=not args.no_charts)
