# Job Posting Velocity as a Lead Indicator for Earnings Surprise Direction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

This repository implements a systematic alternative-data strategy that extracts
forward-looking earnings signals from the *velocity* and *functional composition*
of corporate job postings archived by the Wayback Machine.  The central observation
is that a company's incremental hiring decisions — specifically, whether it is
accelerating headcount in revenue-generating functions (Engineering, Product, Sales)
relative to its own history — precede positive earnings surprises by one to two
quarters.

Building on the labour-economics literature (Lazear & Oyer 2004; Hershbein & Kahn
2018) and the emerging alternative-data signal literature, we construct a composite
signal that combines a rolling z-score of headcount-normalised job-posting velocity
with a functional-decomposition ratio.  We test the signal's predictive content via
Welch t-tests, OLS regression, and Spearman Information Coefficient computation, and
evaluate it in a long-short event-study backtest using the BLS JOLTS openings rate as
a macro regime overlay.

---

## Core Hypothesis

> **H₀**: The mean EPS surprise of companies in the long-signal group (hiring acceleration
> z-score > 1.5 AND signal ratio > 0.65) is equal to the mean EPS surprise of all other
> companies.
>
> **H₁**: The mean EPS surprise of companies in the long-signal group is significantly
> *greater* than that of the control group.

Formally, using a Welch t-test on the distributions of forward surprise percentages:

$$
H_0: \mu_{\text{signal}} = \mu_{\text{control}}
\qquad
H_1: \mu_{\text{signal}} > \mu_{\text{control}}
$$

Rejection at the 5% level is a prerequisite for proceeding to backtest construction.

---

## Methodology

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION LAYER                              │
│                                                                         │
│  Wayback CDX API  ──►  HTML Snapshots  ──►  Job Title Extraction       │
│  (CDX metadata)        (cached .html)        (BeautifulSoup)           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING LAYER                           │
│                                                                         │
│  Functional Decomposition  ──►  Signal Ratio  ──►  Net-New Postings    │
│  (keyword taxonomy)              (bullish ÷ total)  (MoM Δ)            │
│                                                                         │
│  Headcount Normalisation  ──►  Rolling Z-Score  ──►  Composite Signal  │
│  (÷ avg_headcount)              (12-month window)    (long / short)    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     VALIDATION LAYER                                    │
│                                                                         │
│  Signal → Earnings Alignment  ──►  Hypothesis Test  ──►  OLS + IC      │
│  (+1Q / +2Q forward)               (Welch t-test)        (Spearman ρ)  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     BACKTEST LAYER                                      │
│                                                                         │
│  Event Study  ──►  L-S Portfolio  ──►  Performance Metrics             │
│  (hold 60 days)    (tercile rank)       (Sharpe, MaxDD, IC)            │
│                                                                         │
│  JOLTS Macro Overlay  ──►  Regime Attribution                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Signal Construction

The composite long signal requires *both* conditions:

**1. Velocity condition** (z-score threshold):

$$
z_t = \frac{\Delta_t - \bar{\Delta}_{t-12:t}}{\sigma_{\Delta, t-12:t}} > 1.5
$$

where $\Delta_t = \text{net\_new\_postings}_t \,/\, \text{avg\_headcount}$
is the headcount-normalised month-over-month change.

**2. Quality condition** (functional composition threshold):

$$
\text{signal\_ratio}_t = \frac{n_{\text{ENGINEERING}} + n_{\text{PRODUCT}} + n_{\text{SALES}}}{n_{\text{total}}} > 0.65
$$

The **continuous signal strength** used for portfolio ranking is the product:

$$
\text{signal\_strength}_t = z_t \times \text{signal\_ratio}_t
$$

---

## Data Sources

| Source | Access | Description |
|--------|--------|-------------|
| [Wayback Machine CDX API](http://web.archive.org/cdx/search/cdx) | Free, no key | Historical careers page HTML snapshots |
| [yfinance](https://github.com/ranaroussi/yfinance) | Free, no key | EPS estimates, actuals, earnings dates |
| [BLS JOLTS API](https://www.bls.gov/developers/) | Free, registration for v2 | Monthly US job openings rate (JTS000000000000000JOR) |
| `data/reference/company_universe.csv` | Curated | 15-company S&P 500 universe with careers page URLs |
| `data/reference/job_function_taxonomy.csv` | Curated | 70+ keyword → functional category mappings |

---

## Repository Structure

```
job-posting-alpha/
├── README.md                              ← This file
├── requirements.txt                       ← Python dependencies
├── .gitignore                             ← Excludes raw/processed data, outputs
├── config.py                              ← All paths, thresholds, API endpoints
│
├── data/
│   ├── raw/                               ← Cached HTML files (gitignored)
│   ├── processed/                         ← Derived CSVs (gitignored)
│   └── reference/
│       ├── company_universe.csv           ← 15-company scraping universe
│       └── job_function_taxonomy.csv      ← 70+ keyword-to-category mappings
│
├── outputs/                               ← Generated charts (gitignored)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb     ← Full pipeline walkthrough
│
└── src/
    ├── __init__.py
    ├── wayback_scraper.py                 ← CDX API, HTML parsing, timeseries build
    ├── earnings_data.py                   ← yfinance EPS, BLS JOLTS
    ├── features.py                        ← Feature engineering pipeline
    ├── signal.py                          ← Alignment, t-test, OLS, IC
    ├── backtest.py                        ← Event study, L-S backtest, metrics
    └── visualize.py                       ← All 6 publication-quality charts
```

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/your-handle/job-posting-alpha.git
cd job-posting-alpha
pip install -r requirements.txt
```

### 2. Inspect the company universe

```bash
python -c "import pandas as pd, config; print(pd.read_csv(config.COMPANY_UNIVERSE_CSV))"
```

### 3. Smoke test the Wayback scraper (one ticker)

```bash
python -m src.wayback_scraper MSFT
```

This fetches CDX metadata, downloads one snapshot, parses the HTML, and prints
the posting count.  Results are cached so re-runs are instant.

### 4. Build the full universe timeseries

```python
import pandas as pd
import config
from src import wayback_scraper

universe = pd.read_csv(config.COMPANY_UNIVERSE_CSV)
df = wayback_scraper.build_universe_timeseries(universe)
print(df.shape)
```

> ⚠️ This makes live HTTP requests to the Wayback Machine.
> Allow 30–90 minutes for 15 companies × 18 months × 2 snapshots/month.
> All responses are cached; subsequent runs complete in seconds.

### 5. Run the full feature + signal pipeline

```python
from src import features

df = features.compute_functional_decomposition(df)
df = features.compute_net_new_postings(df)
df = features.normalize_by_headcount(df, universe)
df = features.compute_hiring_acceleration_zscore(df)
df = features.build_composite_signal(df)
```

### 6. Fetch earnings and run hypothesis tests

```python
from src import earnings_data, signal

tickers = universe['ticker'].tolist()
earnings_df = earnings_data.fetch_all_earnings(tickers)

aligned = signal.align_signal_to_earnings(df, earnings_df, lead_quarters=1)
stats = signal.run_ttest_validation(aligned)
ic = signal.compute_information_coefficient(aligned)
model = signal.run_ols_regression(aligned)
```

### 7. Run the backtest and generate all charts

```python
from src import backtest, visualize

bt = backtest.run_long_short_backtest(aligned)
metrics = backtest.compute_performance_metrics(bt, aligned)

visualize.plot_functional_decomposition_heatmap(df)
visualize.plot_zscore_distribution(df)
visualize.plot_signal_vs_earnings_surprise(aligned, ic_value=ic)
visualize.plot_cumulative_pnl(bt)
visualize.plot_beat_rate_by_signal_quintile(aligned)
```

All charts are saved to `outputs/` at 300 DPI.

### 8. Open the notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## Key Findings

> **Populate after running the full pipeline with live Wayback + yfinance data.**

Placeholder template:

| Metric | 1Q Lead | 2Q Lead |
|--------|---------|---------|
| t-statistic | TBD | TBD |
| p-value | TBD | TBD |
| Beat rate (signal) | TBD | TBD |
| Beat rate (control) | TBD | TBD |
| Spearman IC | TBD | TBD |
| Annualised L-S Sharpe | TBD | TBD |
| Max Drawdown | TBD | TBD |

---

## Limitations

1. **Wayback Machine HTML rendering**: Modern careers pages rely on JavaScript;
   Wayback archives store the pre-render DOM.  HTML parsing confidence may be
   low for companies using React/Angular SPAs.  Use `extraction_confidence` to
   filter unreliable observations.

2. **Company-level endogeneity**: Companies that anticipate strong earnings may
   proactively increase hiring, creating a feedback loop.  The signal conflates
   *leading* indicators with *coincident* hiring reactions to current business
   conditions.

3. **Survivorship bias**: The current 15-company universe includes only firms
   that are currently large-cap S&P 500 members.  A historical universe would
   need to include de-listed and merged companies to eliminate survivorship bias.

4. **EPS data quality**: yfinance EPS estimates have known quality issues,
   particularly for pre-2015 periods, fiscal year-end companies, and non-US
   reporting standards.  Production use requires a validated consensus source
   (IBES, Refinitiv, Bloomberg).

5. **Sample size constraints**: The 15-company, 18-month universe produces
   O(200) signal events — the minimum for statistical inference.  Conclusions
   drawn from this pilot are directional only; scale to 500+ companies for
   robust out-of-sample validation.

---

## Academic References

- **Lazear, E.P. & Oyer, P. (2004)**. "Internal and External Labor Markets: A Personnel
  Economics Approach." *Labour Economics*, 11(5), 527–554.
  — Foundational paper on job postings as forward-looking firm signals.

- **Hershbein, B. & Kahn, L.B. (2018)**. "Do Recessions Accelerate Routine-Biased
  Technological Change? Evidence from Vacancy Postings." *American Economic Review*,
  108(7), 1737–1772.
  — Shows job posting composition predicts firm-level productivity trajectories.

- **Banker, R.D., Huang, R., Natarajan, R. & Zhao, S. (2019)**. "Market Valuation of
  Intangible Asset Disclosures and Future Earnings." *The Accounting Review*, 94(6).

- **Hou, K., Xue, C. & Zhang, L. (2020)**. "Replicating Anomalies." *Review of
  Financial Studies*, 33(5), 2019–2133.
  — IC threshold benchmarks for alternative-data signals.

- **Kogan, L., Papanikolaou, D., Seru, A. & Stoffman, N. (2017)**. "Technological
  Innovation, Resource Allocation, and Growth." *Quarterly Journal of Economics*,
  132(2), 665–712.

---

## Disclaimer

This repository is intended for **academic research and educational purposes only**.
Nothing in this repository constitutes investment advice, financial advice, or a
recommendation to buy or sell any security.  Past backtest performance does not
guarantee future returns.  All results are produced on synthetic or publicly available
data and have not been validated on live trading infrastructure.

---

*Built with the Wayback Machine CDX API, yfinance, BLS JOLTS, and a lot of BeautifulSoup.*
