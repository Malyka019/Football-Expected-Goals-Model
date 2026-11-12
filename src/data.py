"""Data loading, cleaning, and reshaping for the football goals model.

Pipeline position
-----------------
    results.csv  ->  clean_results()  ->  to_long_format()  ->  features (step 2)

Two shapes of the same data are used throughout the project:

* **match-level** (one row per match): what the Kaggle file gives you, and the
  shape you want for Elo updates and head-to-head history.
* **team-level / long** (two rows per match, one per team): what the models
  consume. A row is "this team, against this opponent, scored this many".

Why long format
---------------
Training a separate home model and away model is the most common mistake in
this kind of project: you get two half-models that can disagree about how good
a team is, and you double the number of bugs. In long format a single model is
symmetric by construction, sees ~2x the rows, and the "which side of the
fixture is this?" question becomes one binary feature (`is_home`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def clean_results(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw Kaggle match table.

    Steps, in order (each one is logged in the returned report):
      1. check the expected columns are present;
      2. parse dates and coerce score columns to numeric;
      3. strip whitespace from team/tournament names (so "Brazil " and
         "Brazil" are the same team *before* duplicate detection);
      4. drop rows without a usable date, team name, or score;
      5. drop exact duplicate fixtures (same day, same two teams);
      6. normalise the `neutral` flag;
      7. add `year`, `tournament_category`, and a stable `match_id`.

    Returns
    -------
    (clean_df, report) -- sorted by date, one row per match, with an integer
    `match_id`. The report is a plain dict so it can be written to JSON.
    """
    report: dict = {"rows_in": int(len(df))}

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"results file is missing required columns: {missing}. "
            f"Expected at least {config.REQUIRED_COLUMNS}."
        )

    out = df.copy()

    # --- 2. types ---------------------------------------------------------
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("home_score", "away_score"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- 3. name normalisation (before dedupe) ----------------------------
    for col in ("home_team", "away_team", "tournament"):
        out[col] = out[col].astype(str).str.strip()

    # --- 4. unusable rows -------------------------------------------------
    # A handful of very old matches have no recorded score. They still carry
    # information for Elo, but for this project they are noise: the target is
    # the score, so a match without one cannot be trained on or evaluated.
    before = len(out)
    out = out.dropna(subset=["date", "home_team", "away_team",
                             "home_score", "away_score"])
    report["dropped_missing_values"] = int(before - len(out))

    # --- 5. duplicates ----------------------------------------------------
    before = len(out)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    report["dropped_duplicates"] = int(before - len(out))

    # `neutral` arrives as real booleans in some exports and as "TRUE"/"FALSE"
    # strings in others, so normalise defensively.
    out["neutral"] = (
        out["neutral"].astype(str).str.strip().str.upper()
        .isin(["TRUE", "1", "1.0", "YES", "T"]).fillna(False).astype(bool)
    )
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)

    # --- 6. derived columns ----------------------------------------------
    out = out[out["date"].dt.year >= config.MIN_YEAR]
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    out["match_id"] = np.arange(len(out), dtype=np.int64)
    out["year"] = out["date"].dt.year.astype(np.int16)
    out["tournament_category"] = out["tournament"].map(config.classify_tournament)

    report["rows_out"] = int(len(out))
    report["date_min"] = out["date"].min().date().isoformat() if len(out) else None
    report["date_max"] = out["date"].max().date().isoformat() if len(out) else None
    report["n_teams"] = int(pd.concat([out["home_team"], out["away_team"]]).nunique())
    report["n_neutral"] = int(out["neutral"].sum())
    report["tournament_categories"] = (
        out["tournament_category"].value_counts().to_dict()
    )
    return out, report


def _apply_former_names(
    df: pd.DataFrame, former_names_csv: Path | None
) -> tuple[pd.DataFrame, int]:
    """Optionally rename defunct team names to their modern successor.

    This is OFF by default, and that is a deliberate modelling choice: folding
    "Soviet Union" into "Russia" or "Yugoslavia" into "Serbia" gives those
    teams a rating history they never had as the same sporting entity. Keep
    historical entities distinct unless you have a reason not to.
    """
    path = Path(former_names_csv) if former_names_csv else config.FORMER_NAMES_CSV
    if not path.exists():
        return df, 0

    mapping = pd.read_csv(path)
    if not {"current", "former"}.issubset(mapping.columns):
        return df, 0
    rename = dict(zip(mapping["former"].str.strip(), mapping["current"].str.strip()))

    out = df.copy()
    out["home_team"] = out["home_team"].replace(rename)
    out["away_team"] = out["away_team"].replace(rename)
    changed = int((out["home_team"] != df["home_team"]).sum()
                  + (out["away_team"] != df["away_team"]).sum())
    return out, changed


def load_results(
    path: str | Path | None = None,
    *,
    rename_former: bool = False,
    min_year: int = config.MIN_YEAR,
) -> pd.DataFrame:
    """Read and clean `results.csv`, returning one row per match.

    Parameters
    ----------
    path
        Location of the Kaggle CSV. Defaults to ``data/raw/results.csv``.
    rename_former
        Merge defunct nations into their successor (see `_apply_former_names`).
    min_year
        Drop matches before this year.
    """
    csv_path = Path(path) if path else config.RESULTS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Download 'International football results from 1872 to present'\n"
            "(Mart Jurisoo, Kaggle) and place results.csv in data/raw/.\n"
            "Or run:  python -m src.pipeline build --synthetic"
        )

    raw = pd.read_csv(csv_path)
    if rename_former:
        raw, _ = _apply_former_names(raw, config.FORMER_NAMES_CSV)

    matches, report = clean_results(raw)
    matches = matches[matches["year"] >= min_year].reset_index(drop=True)
    matches.attrs["cleaning_report"] = report
    matches.attrs["synthetic"] = is_synthetic_data(csv_path)
    return matches


# --------------------------------------------------------------------------
# Reshaping
# --------------------------------------------------------------------------


def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn one-row-per-match into one-row-per-team-per-match.

    Resulting columns
    -----------------
    match_id, date, year, tournament, tournament_category, neutral,
    home_team, away_team,          # fixture identity, handy for joins/display
    team, opponent, is_home,       # this row's perspective
    goals_for, goals_against,      # the two targets
    goal_diff, result,             # "W" / "D" / "L" for this team
    home_advantage                 # 1 only when this team really is at home

    `home_advantage` is separate from `is_home` on purpose: at a neutral venue
    (World Cup, most finals) the fixture still lists a nominal home team, but
    no home advantage applies. Conflating the two is a classic leak of signal
    that does not exist.
    """
    needed = ["match_id", "date", "year", "tournament", "tournament_category",
              "neutral", "home_team", "away_team", "home_score", "away_score"]
    base = matches[needed].copy()

    home = base.assign(
        team=base["home_team"],
        opponent=base["away_team"],
        is_home=1,
        goals_for=base["home_score"],
        goals_against=base["away_score"],
    )
    away = base.assign(
        team=base["away_team"],
        opponent=base["home_team"],
        is_home=0,
        goals_for=base["away_score"],
        goals_against=base["home_score"],
    )

    long = pd.concat([home, away], ignore_index=True)
    long = long.drop(columns=["home_score", "away_score"])

    long["home_advantage"] = (long["is_home"] * (~long["neutral"]).astype(int)).astype(np.int8)
    long["is_home"] = long["is_home"].astype(np.int8)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["result"] = np.select(
        [long["goal_diff"] > 0, long["goal_diff"] == 0],
        ["W", "D"],
        default="L",
    )

    long = long.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False])
    return long.reset_index(drop=True)


def build_datasets(
    *,
    synthetic: bool = False,
    force: bool = False,
    rename_former: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load -> clean -> reshape, writing both parquet files and a JSON report.

    Returns the (match-level, team-level) pair.
    """
    config.ensure_dirs()

    if synthetic and (force or not config.RESULTS_CSV.exists()):
        generate_synthetic_results(config.RESULTS_CSV)
        print(f"[data] wrote synthetic fixture data -> {config.RESULTS_CSV}")

    matches = load_results(rename_former=rename_former)
    long = to_long_format(matches)

    matches.to_parquet(config.MATCHES_CLEAN_PARQUET, index=False)
    long.to_parquet(config.MATCHES_LONG_PARQUET, index=False)

    report = dict(matches.attrs.get("cleaning_report", {}))
    report["long_rows"] = int(len(long))
    report["goals_per_match"] = float(
        (matches["home_score"] + matches["away_score"]).mean()
    )
    report["home_win_share"] = float((matches["home_score"] > matches["away_score"]).mean())
    report["draw_share"] = float((matches["home_score"] == matches["away_score"]).mean())
    (config.INTERIM_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return matches, long


# --------------------------------------------------------------------------
# Synthetic fallback data
# --------------------------------------------------------------------------


def generate_synthetic_results(
    path: str | Path = config.RESULTS_CSV,
    *,
    n_matches: int = 6000,
    seed: int = config.RANDOM_SEED,
) -> Path:
    """Write a fake but structurally realistic `results.csv`.

    The point is *not* to fake good results -- it is to let the whole pipeline
    (cleaning, features, models, evaluation, app) run end to end before the
    real download is in place, and to give the test suite something to chew on.
    Scores are drawn from Poisson rates driven by latent team strength plus a
    home advantage, so downstream models have a real signal to find.
    """
    rng = np.random.default_rng(seed)

    teams = [
        "Argentina", "Brazil", "France", "Germany", "Spain", "England",
        "Italy", "Netherlands", "Portugal", "Belgium", "Croatia", "Uruguay",
        "Colombia", "Mexico", "USA", "Japan", "South Korea", "Morocco",
        "Senegal", "Nigeria", "Egypt", "Australia", "Denmark", "Sweden",
        "Poland", "Serbia", "Switzerland", "Austria", "Turkey", "Ukraine",
        "Wales", "Scotland", "Ireland", "Norway", "Greece", "Czech Republic",
        "Chile", "Peru", "Ecuador", "Paraguay", "Canada", "Costa Rica",
        "Iran", "Saudi Arabia", "Qatar", "Ghana", "Cameroon", "Algeria",
    ]
    attack = dict(zip(teams, rng.normal(0.0, 0.35, len(teams))))
    defence = dict(zip(teams, rng.normal(0.0, 0.30, len(teams))))  # lower is better

    tournaments = [
        ("Friendly", 0.34, 20),
        ("FIFA World Cup qualification", 0.24, 0),
        ("UEFA Euro qualification", 0.10, 0),
        ("FIFA World Cup", 0.06, 0),
        ("UEFA Euro", 0.07, 0),
        ("Copa America", 0.06, 0),
        ("African Cup of Nations", 0.05, 0),
        ("UEFA Nations League", 0.08, 0),
    ]
    names = [t[0] for t in tournaments]
    probs = np.array([t[1] for t in tournaments])
    probs = probs / probs.sum()

    base = np.log(1.35)          # ~1.35 goals per team per match
    home_boost = np.log(1.28)    # ~28% more goals for the home side

    years = rng.integers(1990, 2024, n_matches)

    rows = []
    for year in years:
        i, j = rng.choice(len(teams), size=2, replace=False)
        home_team, away_team = teams[i], teams[j]

        # World Cups only happen every four years and are played at neutral
        # venues, inside a realistic summer (or, for 2022, winter) window.
        if year % 4 == 2 and rng.random() < 0.35:
            tournament = "FIFA World Cup"
            if year == 2022:
                start, end = pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")
            else:
                start, end = pd.Timestamp(f"{year}-06-10"), pd.Timestamp(f"{year}-07-15")
            date = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        else:
            tournament = names[rng.choice(len(names), p=probs)]
            date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(
                days=int(rng.integers(0, 365))
            )
        neutral = tournament == "FIFA World Cup" or rng.random() < 0.08

        lambda_home = np.exp(base + attack[home_team] - defence[away_team]
                             + (0.0 if neutral else home_boost))
        lambda_away = np.exp(base + attack[away_team] - defence[home_team])

        rows.append({
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": int(rng.poisson(lambda_home)),
            "away_score": int(rng.poisson(lambda_away)),
            "tournament": tournament,
            "city": "Somewhere",
            "country": "Nowhere",
            "neutral": bool(neutral),
        })

    synthetic = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(path, index=False)

    # Sidecar marker, so nothing downstream can report fallback numbers as if
    # they were real results.
    Path(str(path) + ".synthetic").write_text(
        "generated by src.data.generate_synthetic_results\n"
    )
    return path


def is_synthetic_data(path: str | Path | None = None) -> bool:
    """True when results.csv came from the synthetic fallback generator."""
    csv_path = Path(path) if path else config.RESULTS_CSV
    return Path(str(csv_path) + ".synthetic").exists()


if __name__ == "__main__":  # tiny manual smoke test
    m, l = build_datasets(synthetic=True)
    print(m.head())
    print(l.head())
