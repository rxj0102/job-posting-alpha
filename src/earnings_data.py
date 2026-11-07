"""
earnings_data.py
================
Fetches historical earnings surprise data via yfinance and macro regime data
from the BLS JOLTS public API.

Functions
---------
fetch_earnings_history   : Pull EPS estimates vs actuals for one ticker.
fetch_all_earnings       : Loop over a ticker list.
fetch_jolts_macro        : Pull BLS JOLTS job-openings-rate series.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Earnings data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_earnings_history(ticker: str) -> pd.DataFrame:
    """
    Pull historical earnings dates, EPS estimates, and actuals via yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``["ticker", "earnings_date", "eps_estimate", "eps_actual",
        "surprise_pct", "beat_flag"]``.
        Saved to ``data/processed/{ticker}_earnings.csv``.
        Returns empty DataFrame if yfinance returns no data.

    Notes
    -----
    - ``surprise_pct`` = ``(eps_actual - eps_estimate) / abs(eps_estimate) * 100``.
      Rows where ``eps_estimate == 0`` are dropped to avoid division errors.
    - ``beat_flag`` is ``True`` when ``eps_actual > eps_estimate``.
    - Results cached to CSV to avoid repeated API calls on re-runs.
    """
    cache_path: Path = config.PROCESSED_DIR / f"{ticker}_earnings.csv"
    if cache_path.exists():
        logger.info("Loading cached earnings for %s from %s", ticker, cache_path)
        return pd.read_csv(cache_path, parse_dates=["earnings_date"])

    logger.info("Fetching earnings history for %s", ticker)
    try:
        yf_ticker = yf.Ticker(ticker)
        earnings_dates = yf_ticker.earnings_dates

        if earnings_dates is None or earnings_dates.empty:
            logger.warning("yfinance returned no earnings dates for %s", ticker)
            return pd.DataFrame(
                columns=["ticker", "earnings_date", "eps_estimate",
                         "eps_actual", "surprise_pct", "beat_flag"]
            )

        df = earnings_dates.reset_index()
        # yfinance column names can vary across versions; normalise them
        col_map: dict[str, str] = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if "date" in lc:
                col_map[col] = "earnings_date"
            elif "estimate" in lc or "expected" in lc:
                col_map[col] = "eps_estimate"
            elif "actual" in lc or "reported" in lc:
                col_map[col] = "eps_actual"
            elif "surprise" in lc and "%" in col:
                col_map[col] = "surprise_pct_raw"
        df = df.rename(columns=col_map)

        # Ensure required columns exist
        for req in ("earnings_date", "eps_estimate", "eps_actual"):
            if req not in df.columns:
                df[req] = float("nan")

        df = df[["earnings_date", "eps_estimate", "eps_actual"]].copy()
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True, errors="coerce")
        df = df.dropna(subset=["earnings_date", "eps_estimate", "eps_actual"])

        # Filter out rows where estimate is zero (avoid ÷0)
        df = df[df["eps_estimate"] != 0].copy()

        df["surprise_pct"] = (
            (df["eps_actual"] - df["eps_estimate"]) / df["eps_estimate"].abs() * 100
        )
        df["beat_flag"] = df["eps_actual"] > df["eps_estimate"]
        df.insert(0, "ticker", ticker)

        # Clip extreme outliers (e.g. when eps_estimate ≈ 0)
        df["surprise_pct"] = df["surprise_pct"].clip(-200, 200)

        df = df.sort_values("earnings_date").reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        logger.info("Saved earnings for %s: %d rows to %s", ticker, len(df), cache_path)
        return df

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch earnings for %s: %s", ticker, exc)
        return pd.DataFrame(
            columns=["ticker", "earnings_date", "eps_estimate",
                     "eps_actual", "surprise_pct", "beat_flag"]
        )


def fetch_all_earnings(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch earnings history for a list of tickers and concatenate into one DataFrame.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.

    Returns
    -------
    pd.DataFrame
        Concatenated earnings DataFrame with the same schema as
        :func:`fetch_earnings_history`.
        Tickers returning empty DataFrames are silently skipped.

    Notes
    -----
    Adds a short sleep between requests to avoid hitting yfinance rate limits.
    """
    frames: list[pd.DataFrame] = []
    n = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        logger.info("Fetching earnings %s (%d/%d)", ticker, i, n)
        df = fetch_earnings_history(ticker)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # light throttle for yfinance

    if not frames:
        logger.error("No earnings data collected for any ticker")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Earnings data: %d rows across %d tickers", len(combined),
                combined["ticker"].nunique())
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# JOLTS macro overlay
# ─────────────────────────────────────────────────────────────────────────────

# Fallback BLS CSV if API is unavailable (public data download)
_JOLTS_FALLBACK_URL = (
    "https://download.bls.gov/pub/time.series/jt/jt.data.1.AllItems"
)


def fetch_jolts_macro(start_year: int = 2010) -> pd.DataFrame:
    """
    Pull the BLS JOLTS Job Openings Rate time-series (monthly).

    Parameters
    ----------
    start_year : int, optional
        First year of data to retrieve (default: 2010).

    Returns
    -------
    pd.DataFrame
        Columns: ``["date", "jolts_openings_rate"]``.
        Saved to ``data/processed/jolts_macro.csv``.

    Notes
    -----
    **Macro Regime Overlay Interpretation**

    A high JOLTS openings rate indicates a tight, broad-based labour market
    in which *most* companies are hiring aggressively.  In this regime,
    a company-level acceleration signal is less differentiating because the
    base rate of hiring is elevated for all firms.

    Conversely, during a period of low JOLTS rates (slack labour market),
    a firm-specific hiring acceleration stands out sharply from the
    cross-sectional distribution and should carry more forward earnings
    information content.

    This series is therefore used as a macro *scaling factor* in the OLS
    regression (``src/signal.py``) and in the performance attribution
    analysis (``src/backtest.py``).

    **API Note**

    BLS v1 API requires no registration key.  BLS v2 (used here) requires a
    free API key for higher request limits; without a key the series data is
    still returned but at a limited rate.  If the API call fails, a fallback
    to the BLS public flat-file download is attempted.
    """
    cache_path: Path = config.JOLTS_MACRO_CSV
    if cache_path.exists():
        logger.info("Loading cached JOLTS macro from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    logger.info("Fetching JOLTS series %s from BLS API", config.JOLTS_SERIES_ID)

    import datetime
    end_year = datetime.datetime.today().year

    # ── Try BLS API ──────────────────────────────────────────────────────────
    payload = {
        "seriesid": [config.JOLTS_SERIES_ID],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    df: Optional[pd.DataFrame] = None  # type: ignore[name-defined]

    try:
        resp = requests.post(
            config.JOLTS_API_URL,
            json=payload,
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "REQUEST_SUCCEEDED":
            series_data = data["Results"]["series"][0]["data"]
            records = []
            for item in series_data:
                # period format: "M01" … "M12"
                period = item["period"]
                if not period.startswith("M") or period == "M13":
                    continue
                month = int(period[1:])
                year = int(item["year"])
                date = pd.Timestamp(year=year, month=month, day=1)
                records.append({"date": date, "jolts_openings_rate": float(item["value"])})
            df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
            logger.info("BLS API returned %d JOLTS observations", len(df))
        else:
            logger.warning("BLS API response status: %s", data.get("status"))

    except Exception as exc:  # noqa: BLE001
        logger.warning("BLS API call failed: %s — trying fallback flat file", exc)

    # ── Fallback: BLS public flat file ───────────────────────────────────────
    if df is None or df.empty:
        logger.info("Attempting JOLTS fallback download from %s", _JOLTS_FALLBACK_URL)
        try:
            resp = requests.get(_JOLTS_FALLBACK_URL, timeout=30)
            resp.raise_for_status()
            raw = pd.read_csv(
                StringIO(resp.text),
                sep=r"\s+",
                header=0,
                on_bad_lines="skip",
            )
            # Filter to target series and job-openings rate
            raw.columns = [c.strip() for c in raw.columns]
            jolts = raw[
                raw["series_id"].str.strip() == config.JOLTS_SERIES_ID
            ].copy()
            jolts["month"] = jolts["period"].str[1:].astype(int)
            jolts["year"] = jolts["year"].astype(int)
            jolts = jolts[jolts["month"] <= 12]
            jolts["date"] = pd.to_datetime(
                jolts[["year", "month"]].assign(day=1).rename(
                    columns={"year": "year", "month": "month"}
                )
            )
            jolts["jolts_openings_rate"] = jolts["value"].astype(float)
            df = (
                jolts[["date", "jolts_openings_rate"]]
                .sort_values("date")
                .reset_index(drop=True)
            )
            df = df[df["date"].dt.year >= start_year]
            logger.info("Flat-file fallback returned %d JOLTS observations", len(df))

        except Exception as exc2:  # noqa: BLE001
            logger.error("Both JOLTS data sources failed: %s", exc2)
            # Return a minimal placeholder so downstream code doesn't crash
            df = pd.DataFrame(columns=["date", "jolts_openings_rate"])

    if df is not None and not df.empty:
        df.to_csv(cache_path, index=False)
        logger.info("JOLTS macro saved to %s", cache_path)

    return df
