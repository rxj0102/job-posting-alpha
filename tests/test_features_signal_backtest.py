"""Offline tests: feature maths, point-in-time behaviour and backtest mechanics."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.backtest import run_event_study, run_long_short_backtest
from src.features import (
    build_composite_signal,
    classify_job_titles,
    compute_functional_decomposition,
    compute_hiring_acceleration_zscore,
    compute_net_new_postings,
    normalize_by_headcount,
)
from src.signal import align_signal_to_earnings


# ----------------------------------------------------------- classification
def test_classify_counts_every_title_exactly_once():
    titles = ["Senior Software Engineer", "Account Executive", "Underwater Basket Weaver"]
    counts = classify_job_titles(titles)
    assert sum(counts.values()) == len(titles)
    assert counts["ENGINEERING"] >= 1
    assert counts["OTHER"] >= 1


def test_functional_decomposition_signal_ratio_and_bad_json():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "year_month": ["202301"] * 3,
            "job_titles_raw": [
                json.dumps(["Software Engineer", "Software Engineer", "Zzz Unknown"]),
                "not-json",
                None,
            ],
        }
    )
    out = compute_functional_decomposition(df)
    assert out["total_count"].tolist() == [3, 0, 0]
    assert out["signal_ratio"].iloc[0] == pytest.approx(2 / 3)
    assert out["signal_ratio"].iloc[1:].isna().all()


# ------------------------------------------------------------ time series
def _panel():
    months = [f"2022{m:02d}" for m in range(1, 13)] + [f"2023{m:02d}" for m in range(1, 7)]
    rng = np.random.default_rng(0)
    rows = []
    for t, base in (("AAA", 100), ("BBB", 5000)):
        counts = base + np.cumsum(rng.integers(-5, 8, len(months)))
        rows += [(t, m, int(c)) for m, c in zip(months, counts)]
    return pd.DataFrame(rows, columns=["ticker", "year_month", "posting_count"])


def test_net_new_is_per_ticker_no_cross_contamination():
    out = compute_net_new_postings(_panel())
    first = out.groupby("ticker").head(1)
    assert first["net_new"].isna().all()
    a = out[out["ticker"] == "AAA"].reset_index(drop=True)
    assert a["net_new"].iloc[1] == a["posting_count"].iloc[1] - a["posting_count"].iloc[0]


def test_headcount_normalisation_handles_missing_and_zero():
    df = compute_net_new_postings(_panel())
    uni = pd.DataFrame({"ticker": ["AAA", "BBB"], "avg_headcount": [1000.0, 0.0]})
    out = normalize_by_headcount(df, uni)
    assert out.loc[out["ticker"] == "BBB", "normalized_posting_rate"].isna().all()
    a = out[out["ticker"] == "AAA"].iloc[1]
    assert a["normalized_posting_rate"] == pytest.approx(a["posting_count"] / 1000.0)


def _accel(df):
    uni = pd.DataFrame({"ticker": ["AAA", "BBB"], "avg_headcount": [1000.0, 50000.0]})
    return compute_hiring_acceleration_zscore(normalize_by_headcount(compute_net_new_postings(df), uni))


def test_acceleration_zscore_is_point_in_time():
    """z at month t must not change when later months are removed."""
    panel = _panel()
    full = _accel(panel).set_index(["ticker", "year_month"])
    cut_months = sorted(panel["year_month"].unique())[:14]
    part = _accel(panel[panel["year_month"].isin(cut_months)]).set_index(["ticker", "year_month"])
    both = part.index
    pd.testing.assert_series_equal(
        part["hiring_accel_zscore"], full.loc[both, "hiring_accel_zscore"], check_names=False
    )


def test_acceleration_zscore_respects_min_periods():
    out = _accel(_panel())
    a = out[out["ticker"] == "AAA"].reset_index(drop=True)
    # net_new is NaN in month 1, so first valid rolling stat needs ZSCORE_MIN_PERIODS obs after that
    assert a["hiring_accel_zscore"].iloc[: config.ZSCORE_MIN_PERIODS].isna().all()


def test_composite_signal_rules():
    df = pd.DataFrame(
        {
            "hiring_accel_zscore": [2.0, 2.0, -2.0, -2.0, np.nan],
            "high_acceleration": [True, True, False, False, False],
            "signal_ratio": [0.9, 0.2, 0.1, 0.9, 0.9],
        }
    )
    out = build_composite_signal(df)
    assert out["long_signal"].tolist() == [True, False, False, False, False]
    assert out["short_signal"].tolist() == [False, False, True, False, False]
    assert not (out["long_signal"] & out["short_signal"]).any()


# --------------------------------------------------------------- alignment
def test_alignment_only_matches_earnings_after_the_signal_month():
    signal = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "year_month": ["202301"],
            "long_signal": [True],
            "short_signal": [False],
            "signal_strength": [1.0],
            "hiring_accel_zscore": [2.0],
            "signal_ratio": [0.8],
        }
    )
    earnings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "earnings_date": pd.to_datetime(["2022-11-01", "2023-02-10", "2023-04-25"]),
            "beat_flag": [0, 1, 0],
            "surprise_pct": [-1.0, 5.0, 2.0],
        }
    )
    out = align_signal_to_earnings(signal, earnings)
    # window starts 30 days after Jan-31: the Feb-10 print is too early, the Nov print is in the past
    assert len(out) == 1
    assert out["earnings_date"].iloc[0] == pd.Timestamp("2023-04-25")
    assert (out["earnings_date"] > pd.Timestamp("2023-01-31")).all()


# ----------------------------------------------------------------- backtest
def _prices():
    idx = pd.bdate_range("2023-01-02", "2023-06-30")
    return pd.DataFrame(
        {"ticker": "AAA", "date": idx, "close": 100 * 1.001 ** np.arange(len(idx))}
    )


def test_event_study_enters_at_or_before_month_end_and_flips_short():
    px = _prices()
    aligned = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "signal_month": ["202301", "202301"],
            "long_signal": [True, False],
            "short_signal": [False, True],
            "signal_strength": [1.0, -1.0],
            "beat_flag": [1, 0],
            "surprise_pct": [3.0, -3.0],
        }
    )
    res = run_event_study(aligned, px, holding_days=30)
    assert (res["entry_date"] <= pd.Timestamp("2023-01-31")).all()
    assert (res["exit_date"] > res["entry_date"]).all()
    lg = res[res["signal_type"] == "LONG"].iloc[0]
    sh = res[res["signal_type"] == "SHORT"].iloc[0]
    assert lg["holding_return"] > 0 > sh["holding_return"]  # rising price: long wins, short loses
    assert lg["holding_return"] == pytest.approx(-sh["holding_return"])


def test_event_study_ignores_prices_before_entry():
    px = _prices()
    aligned = pd.DataFrame(
        {"ticker": ["AAA"], "signal_month": ["202302"], "long_signal": [True], "short_signal": [False],
         "signal_strength": [1.0], "beat_flag": [1], "surprise_pct": [1.0]}
    )
    base = run_event_study(aligned, px, 30)["holding_return"].iloc[0]
    px2 = px.copy()
    px2.loc[px2["date"] < "2023-02-28", "close"] *= 7
    entry_basis = run_event_study(aligned, px2, 30)["holding_return"].iloc[0]
    assert entry_basis == pytest.approx(base)


def test_long_short_backtest_orders_by_signal_strength():
    rows = []
    for m in ("202301", "202302", "202303"):
        for i in range(6):
            rows.append(("T%d" % i, m, float(i), float(i), i >= 4, i <= 1))  # strength == surprise
    aligned = pd.DataFrame(
        rows, columns=["ticker", "signal_month", "signal_strength", "surprise_pct", "long_signal", "short_signal"]
    )
    res = run_long_short_backtest(aligned)
    assert len(res) == 3
    assert (res["ls_return"] > 0).all()
    assert (res["n_long"] == 2).all() and (res["n_short"] == 2).all()
    assert res["cumulative_pnl"].iloc[-1] > 0


def test_long_short_backtest_needs_three_months():
    aligned = pd.DataFrame(
        {"ticker": list("ABC"), "signal_month": ["202301"] * 3, "signal_strength": [1.0, 0.0, -1.0],
         "surprise_pct": [1.0, 0.0, -1.0], "long_signal": [True, False, False], "short_signal": [False, False, True]}
    )
    assert run_long_short_backtest(aligned).empty
