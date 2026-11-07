"""
config.py
=========
Central configuration for the Job Posting Velocity strategy research pipeline.

All paths, API endpoints, thresholds, and keyword lists live here so that
no module contains hard-coded values.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root (always the directory containing this file)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Data directories
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REFERENCE_DIR: Path = DATA_DIR / "reference"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Ensure directories exist at import time
for _d in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reference file paths
# ─────────────────────────────────────────────────────────────────────────────
COMPANY_UNIVERSE_CSV: Path = REFERENCE_DIR / "company_universe.csv"
JOB_TAXONOMY_CSV: Path = REFERENCE_DIR / "job_function_taxonomy.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Processed data file paths
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_TIMESERIES_CSV: Path = PROCESSED_DIR / "universe_timeseries.csv"
JOLTS_MACRO_CSV: Path = PROCESSED_DIR / "jolts_macro.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Wayback Machine CDX API
# ─────────────────────────────────────────────────────────────────────────────
WAYBACK_CDX_URL: str = "http://web.archive.org/cdx/search/cdx"
WAYBACK_BASE_URL: str = "http://web.archive.org/web"

# ─────────────────────────────────────────────────────────────────────────────
# BLS JOLTS API
# ─────────────────────────────────────────────────────────────────────────────
JOLTS_API_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
JOLTS_SERIES_ID: str = "JTS000000000000000JOR"  # Total US job openings rate

# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_MONTHS: int = 18
EARNINGS_LEAD_QUARTERS: list[int] = [1, 2]  # Test 1Q-ahead and 2Q-ahead

HIRING_ACCEL_ZSCORE_THRESHOLD: float = 1.5
SIGNAL_RATIO_THRESHOLD: float = 0.65

# Rolling window (months) for z-score normalization
ZSCORE_ROLLING_WINDOW: int = 12

# Minimum data points required before computing z-score
ZSCORE_MIN_PERIODS: int = 6

# ─────────────────────────────────────────────────────────────────────────────
# Job title keyword lists
# ─────────────────────────────────────────────────────────────────────────────
ENGINEERING_KEYWORDS: list[str] = [
    "engineer",
    "developer",
    "scientist",
    "architect",
    "product manager",
    "data",
    "research",
    "machine learning",
    "software",
    "infrastructure",
]

GA_KEYWORDS: list[str] = [
    "legal",
    "counsel",
    "compliance",
    "hr",
    "recruiting",
    "finance",
    "accounting",
    "administrative",
    "operations",
    "facilities",
]

# ─────────────────────────────────────────────────────────────────────────────
# HTTP / rate-limiting
# ─────────────────────────────────────────────────────────────────────────────
REQUEST_DELAY_SECONDS: float = 2.0   # Respectful delay between Wayback calls
HTTP_TIMEOUT_SECONDS: int = 30       # Per-request timeout
HTTP_MAX_RETRIES: int = 3            # Exponential-backoff retry attempts

# ─────────────────────────────────────────────────────────────────────────────
# Backtest parameters
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_HOLDING_DAYS: int = 60

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
FIGURE_DPI: int = 300
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"

# ─────────────────────────────────────────────────────────────────────────────
# Logging format
# ─────────────────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
