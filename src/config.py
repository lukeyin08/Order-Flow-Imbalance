"""Shared configuration for the Order Flow Imbalance project.

All magic numbers live here so the analysis is reproducible and easy to tweak.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
REPORT_DIR = os.path.join(ROOT, "report")

for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 20260614

# ---------------------------------------------------------------------------
# Instrument / market conventions
#   We model a liquid crypto perp-style instrument priced around $100 for
#   readability. TICK is the minimum price increment.
# ---------------------------------------------------------------------------
TICK = 0.01            # price increment in dollars
INIT_PRICE = 100.00    # starting mid price
LOT = 1.0              # one unit of size

# Trading costs for the *taker* backtest
TAKER_FEE_BPS = 0.5    # exchange taker fee, in basis points of notional (per side)
MAKER_FEE_BPS = 0.0    # maker fee (often a rebate on real venues; 0 here = conservative)

# ---------------------------------------------------------------------------
# Bucketing
#   Events are aggregated into fixed-length time buckets for the regressions
#   and backtest. 1.0 second is a standard intraday horizon for OFI studies.
# ---------------------------------------------------------------------------
BUCKET_SECONDS = 1.0
FORECAST_HORIZONS = [1, 2, 3, 5, 10]   # in buckets

# Annualisation for a 24/7 crypto market
SECONDS_PER_YEAR = 365 * 24 * 3600
BUCKETS_PER_YEAR = SECONDS_PER_YEAR / BUCKET_SECONDS

# ---------------------------------------------------------------------------
# Simulator size
# ---------------------------------------------------------------------------
N_EVENTS = 300_000     # number of order-book events to simulate
N_LEVELS = 5           # depth levels tracked per side (for multi-level OFI)
