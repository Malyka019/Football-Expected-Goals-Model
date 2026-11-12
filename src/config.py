"""Project-wide configuration: paths, split dates, and tournament importance.

Everything that a downstream module might want to tune lives here, so that
"what is a World Cup qualifier worth?" is answered in exactly one place.

Note on tournament importance
-----------------------------
Two things use it:
  1. the Elo update (step 2), where it sets the K-factor;
  2. the model (steps 4-5), where it is a feature / sample weight.
Keeping a single source of truth avoids the classic bug where the Elo
K-table and the model weight table drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_CSV = RAW_DIR / "results.csv"
FORMER_NAMES_CSV = RAW_DIR / "former_names.csv"

MATCHES_CLEAN_PARQUET = INTERIM_DIR / "matches_clean.parquet"
MATCHES_LONG_PARQUET = PROCESSED_DIR / "matches_long.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

ALL_DIRS = [
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR,
]


def ensure_dirs() -> None:
    """Create the project's data/report directories if they do not exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# Matches before this year are kept for rating history but are usually too
# different (and too sparse) to be worth training on.
MIN_YEAR = 1872

# A single match's columns in the Kaggle `results.csv`.
RESULTS_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]
REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "neutral",
]

# --------------------------------------------------------------------------
# Time-based splits (never shuffle: a random split leaks the future)
# --------------------------------------------------------------------------
TRAIN_END = "2017-12-31"          # everything up to here is training data
VAL_START, VAL_END = "2018-01-01", "2021-12-31"   # includes the 2018 World Cup
TEST_START, TEST_END = "2022-01-01", "2022-12-31"  # includes the 2022 World Cup

# Exact tournament windows, used for the headline "World Cup only" evaluation.
WORLD_CUP_WINDOWS = {
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}

# --------------------------------------------------------------------------
# Tournament categories and importance weights
# --------------------------------------------------------------------------
# The Kaggle `tournament` column has ~150 distinct free-text values
# ("FIFA World Cup", "UEFA Euro qualification", "Copa América", ...).
# Collapsing them into a handful of categories keeps the feature space small
# and the Elo K-factor table readable.
TOURNAMENT_CATEGORIES = [
    "world_cup",
    "continental_final",
    "qualifier",
    "nations_league",
    "friendly",
    "other",
]

# Importance in [0, 1]. Used as: K_effective = K_BASE * importance
# (World Football Elo Ratings uses a similar 20..60 scale).
IMPORTANCE_WEIGHTS = {
    "world_cup": 1.00,
    "continental_final": 0.85,
    "qualifier": 0.75,
    "nations_league": 0.60,
    "other": 0.50,
    "friendly": 0.35,
}

_QUALIFIER_RE = re.compile(r"qualif", re.IGNORECASE)
# "championship" is deliberately absent: "British Home Championship" is not a
# continental final, while the real continental events are covered by name.
_CONTINENTAL_RE = re.compile(
    r"euro|copa|cup of nations|asian cup|gold cup|oceania|confederations|"
    r"african nations|caribbean",
    re.IGNORECASE,
)


def classify_tournament(tournament: str) -> str:
    """Map a free-text tournament name onto one of TOURNAMENT_CATEGORIES.

    Order matters: "FIFA World Cup qualification" contains "world cup" but is
    a qualifier, so the qualifier test has to come first.
    """
    text = str(tournament or "").strip()

    if _QUALIFIER_RE.search(text):
        return "qualifier"
    if "world cup" in text.lower():
        return "world_cup"
    if "nations league" in text.lower():
        return "nations_league"
    if _CONTINENTAL_RE.search(text):
        return "continental_final"
    if "friendly" in text.lower():
        return "friendly"
    return "other"


def tournament_importance(category: str) -> float:
    """Importance weight for a tournament category (defaults to 0.5)."""
    return IMPORTANCE_WEIGHTS.get(category, IMPORTANCE_WEIGHTS["other"])
