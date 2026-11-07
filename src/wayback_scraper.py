"""
wayback_scraper.py
==================
Data acquisition via the Wayback Machine CDX API.

Pipeline
--------
1. Query CDX API for available snapshots of a careers page URL.
2. For each snapshot, fetch the archived HTML (with caching and retry logic).
3. Parse HTML to count job postings and extract job title text.
4. Aggregate into a monthly time-series per company.
5. Build universe-wide time-series DataFrame.

Usage (standalone smoke test)
------------------------------
    python -m src.wayback_scraper
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CDX Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

def get_wayback_snapshots(
    url: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Query the Wayback Machine CDX API and return available snapshots.

    Parameters
    ----------
    url : str
        The careers page URL to look up (e.g. ``https://careers.microsoft.com``).
    from_date : str
        Start date in ``YYYYMMDD`` format (e.g. ``"20220101"``).
    to_date : str
        End date in ``YYYYMMDD`` format (e.g. ``"20231231"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``["timestamp", "snapshot_url"]``.
        Collapsed to at most 2 snapshots per calendar month (first + last available).
        Empty DataFrame if no snapshots are found.

    Notes
    -----
    - Uses ``collapse=digest`` to de-duplicate near-identical snapshots.
    - Logs each CDX request with timestamp and URL for audit purposes.
    - Respects ``config.REQUEST_DELAY_SECONDS`` between network calls.
    """
    params: dict = {
        "url": url,
        "output": "json",
        "from": from_date,
        "to": to_date,
        "fl": "timestamp,statuscode",
        "filter": "statuscode:200",
        "collapse": "digest",
        "limit": 1000,
    }

    logger.info("CDX query | url=%s | from=%s | to=%s", url, from_date, to_date)
    time.sleep(config.REQUEST_DELAY_SECONDS)

    try:
        resp = requests.get(
            config.WAYBACK_CDX_URL,
            params=params,
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("CDX request failed for %s: %s", url, exc)
        return pd.DataFrame(columns=["timestamp", "snapshot_url"])

    data = resp.json()
    if not data or len(data) <= 1:  # CDX returns a header row + data rows
        logger.warning("No snapshots found for %s between %s and %s", url, from_date, to_date)
        return pd.DataFrame(columns=["timestamp", "snapshot_url"])

    # First row is the field header from CDX
    header, *rows = data
    if not rows:
        return pd.DataFrame(columns=["timestamp", "snapshot_url"])

    df = pd.DataFrame(rows, columns=header)

    # Build snapshot URLs
    df["snapshot_url"] = df["timestamp"].apply(
        lambda ts: f"{config.WAYBACK_BASE_URL}/{ts}/{url}"
    )

    # Parse timestamp into datetime for grouping
    df["dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["year_month"] = df["dt"].dt.to_period("M")

    # Keep at most 2 snapshots per month: first and last of each month
    def _first_last(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("dt")
        if len(group) == 1:
            return group.iloc[[0]]
        return pd.concat([group.iloc[[0]], group.iloc[[-1]]])

    collapsed = (
        df.groupby("year_month", group_keys=False)
        .apply(_first_last)
        .reset_index(drop=True)
    )

    logger.info(
        "CDX result | url=%s | raw_snapshots=%d | collapsed=%d",
        url,
        len(df),
        len(collapsed),
    )
    return collapsed[["timestamp", "snapshot_url"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# HTML fetching with caching and retry
# ─────────────────────────────────────────────────────────────────────────────

def fetch_snapshot_html(snapshot_url: str) -> str:
    """
    Fetch a single archived HTML page from the Wayback Machine.

    Parameters
    ----------
    snapshot_url : str
        A fully formed Wayback snapshot URL
        (e.g. ``http://web.archive.org/web/20230315120000/https://careers.microsoft.com``).

    Returns
    -------
    str
        Raw HTML content of the archived page. Empty string on persistent failure.

    Notes
    -----
    - Responses are cached to ``data/raw/`` using an MD5 hash of the URL as the filename.
      Subsequent calls for the same URL return the cached copy immediately.
    - Implements exponential back-off: delays of 2^1, 2^2, 2^3 seconds on retry.
    - Respects ``config.REQUEST_DELAY_SECONDS`` between fresh network requests.
    """
    url_hash = hashlib.md5(snapshot_url.encode()).hexdigest()
    cache_path: Path = config.RAW_DIR / f"{url_hash}.html"

    if cache_path.exists():
        logger.debug("Cache hit: %s", snapshot_url)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    html = ""
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        try:
            logger.info("Fetching snapshot (attempt %d): %s", attempt, snapshot_url)
            time.sleep(config.REQUEST_DELAY_SECONDS)
            resp = requests.get(
                snapshot_url,
                timeout=config.HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": "research-bot/1.0 (academic use)"},
            )
            resp.raise_for_status()
            html = resp.text
            break
        except requests.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(
                "Fetch failed (attempt %d/%d) for %s: %s — retrying in %ds",
                attempt,
                config.HTTP_MAX_RETRIES,
                snapshot_url,
                exc,
                wait,
            )
            time.sleep(wait)

    if html:
        cache_path.write_text(html, encoding="utf-8", errors="replace")
        logger.debug("Cached %d bytes to %s", len(html), cache_path)
    else:
        logger.error("All retries exhausted for %s", snapshot_url)

    return html


# ─────────────────────────────────────────────────────────────────────────────
# HTML parsing
# ─────────────────────────────────────────────────────────────────────────────

# CSS class substrings that typically indicate a job listing container
_JOB_CLASS_HINTS: tuple[str, ...] = (
    "job",
    "position",
    "opening",
    "role",
    "listing",
    "vacancy",
    "career",
    "requisition",
    "posting",
)


def count_job_postings_from_html(html: str, company_name: str) -> dict:
    """
    Parse archived careers-page HTML and estimate the number of open roles.

    Parameters
    ----------
    html : str
        Raw HTML string from a Wayback snapshot.
    company_name : str
        Used only for logging / debug messages.

    Returns
    -------
    dict
        ``{``
        ``  "posting_count": int,``
        ``  "job_titles": list[str],``
        ``  "extraction_confidence": float``
        ``}``

        ``extraction_confidence`` legend:

        - ``1.0`` — a known structural pattern matched (class-name heuristic).
        - ``0.5`` — fallback ``<li>`` heuristic used.
        - ``0.0`` — no jobs detected.

    Notes
    -----
    Strategy is a best-effort multi-pass heuristic:

    1. Look for ``<li>`` or ``<div>`` elements whose ``class`` attribute contains
       any of ``_JOB_CLASS_HINTS``.
    2. Fallback: count ``<li>`` items inside a ``<ul>`` with ≥ 3 children
       that look like text titles (> 10 chars, ≤ 120 chars).
    3. Attempt to extract visible job title text from matched nodes.
    """
    if not html:
        return {"posting_count": 0, "job_titles": [], "extraction_confidence": 0.0}

    soup = BeautifulSoup(html, "lxml")

    # Remove navigation, footer, script, style noise
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    job_nodes: list = []
    confidence = 0.0

    # ── Pass 1: class-name heuristic ────────────────────────────────────────
    for tag in soup.find_all(["li", "div", "article", "tr"]):
        classes = " ".join(tag.get("class", [])).lower()
        if any(hint in classes for hint in _JOB_CLASS_HINTS):
            job_nodes.append(tag)

    if job_nodes:
        confidence = 1.0
    else:
        # ── Pass 2: <li> structural fallback ────────────────────────────────
        for ul in soup.find_all("ul"):
            items = ul.find_all("li", recursive=False)
            if len(items) < 3:
                continue
            candidate_titles = [
                li.get_text(strip=True)
                for li in items
                if 10 < len(li.get_text(strip=True)) <= 120
            ]
            if len(candidate_titles) >= 3:
                job_nodes.extend(items)

        if job_nodes:
            confidence = 0.5

    # De-duplicate nodes (a parent may have been collected alongside its child)
    seen_texts: set[str] = set()
    job_titles: list[str] = []
    for node in job_nodes:
        text = node.get_text(separator=" ", strip=True)
        # Filter out obviously non-title text (too short or too long)
        if len(text) < 5 or len(text) > 200:
            continue
        if text not in seen_texts:
            seen_texts.add(text)
            job_titles.append(text)

    posting_count = len(job_titles)

    if posting_count == 0:
        confidence = 0.0

    logger.debug(
        "Parsed %s: count=%d, confidence=%.1f",
        company_name,
        posting_count,
        confidence,
    )
    return {
        "posting_count": posting_count,
        "job_titles": job_titles,
        "extraction_confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-company pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_posting_timeseries(
    ticker: str,
    careers_url: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Build a monthly job-posting time-series for a single company.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"MSFT"``).
    careers_url : str
        The company careers page URL to track.
    from_date : str
        Start date ``YYYYMMDD``.
    to_date : str
        End date ``YYYYMMDD``.

    Returns
    -------
    pd.DataFrame
        Columns: ``["ticker", "year_month", "posting_count",
        "job_titles_raw", "extraction_confidence"]``.
        Saved to ``data/processed/{ticker}_timeseries.csv``.
        Returns empty DataFrame if no snapshots found.

    Notes
    -----
    Each snapshot's parsed results are averaged within a calendar month
    (taking the mean posting_count and the minimum extraction_confidence
    as a conservative quality indicator).
    """
    snapshots_df = get_wayback_snapshots(careers_url, from_date, to_date)
    if snapshots_df.empty:
        logger.warning("No snapshots for %s — returning empty DataFrame", ticker)
        return pd.DataFrame(
            columns=["ticker", "year_month", "posting_count", "job_titles_raw",
                     "extraction_confidence"]
        )

    records: list[dict] = []
    for _, row in snapshots_df.iterrows():
        html = fetch_snapshot_html(row["snapshot_url"])
        result = count_job_postings_from_html(html, ticker)

        # Derive year_month from CDX timestamp (first 6 chars = YYYYMM)
        year_month = str(row["timestamp"])[:6]

        records.append(
            {
                "ticker": ticker,
                "snapshot_timestamp": row["timestamp"],
                "year_month": year_month,
                "posting_count": result["posting_count"],
                "job_titles_raw": json.dumps(result["job_titles"]),
                "extraction_confidence": result["extraction_confidence"],
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["ticker", "year_month", "posting_count",
                     "job_titles_raw", "extraction_confidence"]
        )

    raw_df = pd.DataFrame(records)

    # Collapse to monthly by averaging count, min confidence (conservative)
    monthly = (
        raw_df.groupby(["ticker", "year_month"])
        .agg(
            posting_count=("posting_count", "mean"),
            job_titles_raw=("job_titles_raw", "first"),  # keep first snapshot's titles
            extraction_confidence=("extraction_confidence", "min"),
        )
        .reset_index()
    )

    out_path: Path = config.PROCESSED_DIR / f"{ticker}_timeseries.csv"
    monthly.to_csv(out_path, index=False)
    logger.info("Saved %s timeseries to %s (%d rows)", ticker, out_path, len(monthly))

    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Universe-wide pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_universe_timeseries(company_universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build job-posting time-series for every company in the universe CSV.

    Parameters
    ----------
    company_universe_df : pd.DataFrame
        Must contain columns ``["ticker", "careers_page_url"]``.

    Returns
    -------
    pd.DataFrame
        Concatenated monthly time-series for all companies.
        Saved to ``data/processed/universe_timeseries.csv``.

    Notes
    -----
    - Progress is logged as ``"Processing {ticker} ({i}/{n})"``.
    - Individual per-company CSVs are also written by :func:`build_posting_timeseries`.
    - Companies that return empty DataFrames are skipped with a warning.
    """
    from datetime import datetime, timedelta

    to_date = datetime.today().strftime("%Y%m%d")
    from_dt = datetime.today() - timedelta(days=config.LOOKBACK_MONTHS * 30)
    from_date = from_dt.strftime("%Y%m%d")

    all_frames: list[pd.DataFrame] = []
    n = len(company_universe_df)

    for i, row in enumerate(company_universe_df.itertuples(), start=1):
        ticker = row.ticker
        url = row.careers_page_url
        logger.info("Processing %s (%d/%d)", ticker, i, n)

        df = build_posting_timeseries(ticker, url, from_date, to_date)
        if df.empty:
            logger.warning("Skipping %s — no data returned", ticker)
        else:
            all_frames.append(df)

    if not all_frames:
        logger.error("No data collected for any ticker in the universe")
        return pd.DataFrame()

    universe_df = pd.concat(all_frames, ignore_index=True)
    universe_df.to_csv(config.UNIVERSE_TIMESERIES_CSV, index=False)
    logger.info(
        "Universe timeseries saved to %s (%d rows, %d tickers)",
        config.UNIVERSE_TIMESERIES_CSV,
        len(universe_df),
        universe_df["ticker"].nunique(),
    )
    return universe_df


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Smoke test on a single ticker
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    test_url = "https://careers.microsoft.com/us/en"

    print(f"\n{'='*60}")
    print(f"Smoke test: {test_ticker} — {test_url}")
    print(f"{'='*60}\n")

    # Test CDX lookup (short window to keep it fast)
    snapshots = get_wayback_snapshots(test_url, "20230101", "20230630")
    print(f"Found {len(snapshots)} snapshots:\n{snapshots.head(5)}\n")

    if not snapshots.empty:
        first_url = snapshots.iloc[0]["snapshot_url"]
        html = fetch_snapshot_html(first_url)
        result = count_job_postings_from_html(html, test_ticker)
        print(f"Parsed result: count={result['posting_count']}, "
              f"confidence={result['extraction_confidence']:.1f}")
        print(f"Sample titles: {result['job_titles'][:5]}")
