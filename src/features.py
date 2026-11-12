"""Point-in-time feature engineering (step 2).

The single rule this module obeys
---------------------------------
**A feature for a match may only use information available before kick-off.**

That is easy to say and surprisingly easy to break. Three common ways it breaks,
all of which this module avoids:

1. `groupby(...).mean()` over a team's whole history -- a rolling mean must be
   shifted by one row so the current match is not inside its own window.
2. A "league average" computed over the whole dataset -- must be an *expanding*
   average over previous dates only.
3. Ratings updated in place and then read for the same match -- must store the
   pre-match value and update afterwards.

`tests/test_no_leakage.py` verifies all three by truncating the dataset at
various dates and asserting the surviving rows are byte-identical.

Features produced
-----------------
Elo            elo_team, elo_opponent, elo_diff
Form           form_gf_5, form_ga_5, form_points_5, form_win_5  (and _10)
Strength       attack, defence   (relative to the time-varying league average,
                                  adjusted for the strength of opponents faced)
Context        home_advantage, neutral, tournament_category
History        h2h_played, h2h_gd, h2h_win
Bookkeeping    history_matches   (how many matches the form windows are based on)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Elo constants
# --------------------------------------------------------------------------
# Defaults follow the World Football Elo Ratings convention: 1500 for a new
# team, 100 rating points of home advantage, and a base K of 60 scaled by
# tournament importance (so a World Cup is K=60 and a friendly is K=21).
START_RATING = 1500.0
HOME_ADVANTAGE_ELO = 100.0
K_BASE = 60.0

# Feature-engineering knobs
FORM_WINDOWS = (5, 10)
STRENGTH_HALFLIFE = 20       # matches; recency decay for attack/defence
STRENGTH_MIN_PERIODS = 3     # below this we fall back to "average"
# Exponent on the opponent adjustment. 1.0 = full (undamped) adjustment, 0.0 =
# none. Damped at 0.5 on purpose: a full adjustment double-counts, because the
# opponent's conceding ratio already contains the goals this team scored past
# them, so with little history it cancels the signal it was meant to correct.
OPPONENT_ADJUSTMENT = 0.5
LEAGUE_PRIOR = 1.35          # goals per team per match, used before any history
H2H_RECENT_N = 10            # most recent meetings used for the h2h features


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------


def goal_difference_multiplier(goal_diff: int) -> float:
    """Weight a result by how emphatic it was (World Football Elo style).

    A one-goal win moves ratings by 1.0x, a two-goal win by 1.5x, and a
    three-goal win by 1.75x rising to 2.0x for six goals -- so thrashings are
    not treated as equivalent to narrow wins.
    """
    d = abs(int(goal_diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """Logistic expectation that A beats B (draws count as half a win)."""
    diff = rating_a - rating_b + home_advantage
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def add_elo_ratings(
    matches: pd.DataFrame,
    *,
    start_rating: float = START_RATING,
    k_base: float = K_BASE,
    home_advantage: float = HOME_ADVANTAGE_ELO,
) -> pd.DataFrame:
    """Attach pre-match and post-match Elo ratings to a match-level table.

    The loop is deliberately explicit rather than vectorised: Elo is a recursive
    filter, each update depends on the previous one, and a readable 25-line loop
    that a reviewer can check is worth more than a clever 6-line cumsum.

    Returns the input sorted by (date, match_id) with four new columns:
    elo_home_pre, elo_away_pre, elo_home_post, elo_away_post.
    """
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    n = len(df)
    pre_home = np.empty(n, dtype=float)
    pre_away = np.empty(n, dtype=float)
    post_home = np.empty(n, dtype=float)
    post_away = np.empty(n, dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, start_rating)
        r_away = ratings.get(away, start_rating)

        # Record BEFORE updating: this is the feature the model will see.
        pre_home[i] = r_home
        pre_away[i] = r_away

        # Home advantage only exists when the venue is not neutral.
        bonus = 0.0 if bool(row.neutral) else home_advantage
        we_home = expected_score(r_home, r_away, bonus)

        if row.home_score > row.away_score:
            actual = 1.0
        elif row.home_score == row.away_score:
            actual = 0.5
        else:
            actual = 0.0

        k = k_base * config.tournament_importance(row.tournament_category)
        g = goal_difference_multiplier(row.home_score - row.away_score)
        delta = k * g * (actual - we_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
        post_home[i] = ratings[home]
        post_away[i] = ratings[away]

    df["elo_home_pre"] = pre_home
    df["elo_away_pre"] = pre_away
    df["elo_home_post"] = post_home
    df["elo_away_post"] = post_away
    return df


# --------------------------------------------------------------------------
# Rolling form (team-level, long format)
# --------------------------------------------------------------------------


def add_rolling_form(
    long: pd.DataFrame, windows: tuple[int, ...] = FORM_WINDOWS
) -> pd.DataFrame:
    """Add rolling goals-for / goals-against / points / win-rate over N matches.

    `shift(1)` is the whole trick: the window for a match covers the team's
    previous N matches, never the current one. `min_periods` stays low so that
    a team with two matches of history still gets a (noisy) value rather than
    NaN -- `history_matches` tells the model how much to trust it.
    """
    df = long.sort_values(["team", "date", "match_id"]).copy()

    df["points"] = np.where(df["result"] == "W", 3, np.where(df["result"] == "D", 1, 0))
    df["is_win"] = (df["result"] == "W").astype(float)

    grouped = df.groupby("team", sort=False)
    df["history_matches"] = grouped.cumcount()

    for w in windows:
        for source, name in (
            ("goals_for", "form_gf"),
            ("goals_against", "form_ga"),
            ("points", "form_points"),
            ("is_win", "form_win"),
        ):
            df[f"{name}_{w}"] = grouped[source].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

    return df


# --------------------------------------------------------------------------
# Attack / defence strength
# --------------------------------------------------------------------------


def _league_average_by_date(matches: pd.DataFrame) -> pd.Series:
    """Goals per team per match, expanding over previous dates only.

    Same-day matches are excluded from each other's average: a match on
    2018-06-16 must not see the other games played that afternoon.
    """
    per_date = (
        matches.assign(total=matches["home_score"] + matches["away_score"])
        .groupby("date")
        .agg(total=("total", "sum"), n=("match_id", "size"))
        .sort_index()
    )
    cum_goals = per_date["total"].cumsum().shift(1)
    cum_matches = per_date["n"].cumsum().shift(1)
    # 2 team-rows (goals-for summed once per side) per match.
    league_avg = (cum_goals / (2.0 * cum_matches)).fillna(LEAGUE_PRIOR)
    league_avg.name = "league_avg_goals"
    return league_avg


def add_strength(long: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attack and defence strength relative to the time-varying league average.

    Construction (two passes, both strictly backwards-looking):

    1. **Raw ratio.** A team's exponentially weighted goals scored (and
       conceded) divided by the league average for that date. 1.0 = average.
    2. **Opponent adjustment.** A team that has only played weak sides looks
       strong in pass 1. So divide by what the opponent normally concedes
       (attack) or normally scores (defence) -- raised to a damped power
       (OPPONENT_ADJUSTMENT), because a full-strength single-step adjustment
       double-counts in small samples. This is one Bradley-Terry style update,
       not a fixed point; the joint, iterative fit is what the Dixon-Coles
       model in step 4 does properly.

    All values are clipped into [0.2, 5.0] so a team with 3 matches of history
    cannot blow up a regression.
    """
    league_avg = _league_average_by_date(matches)
    df = long.merge(
        league_avg.rename_axis("date").reset_index(), on="date", how="left"
    )
    df["league_avg_goals"] = df["league_avg_goals"].fillna(LEAGUE_PRIOR)

    # --- pass 1: raw, recency-weighted, point-in-time ---------------------
    df = df.sort_values(["team", "date", "match_id"]).copy()
    grouped = df.groupby("team", sort=False)
    for source, name in (("goals_for", "gf_ewm"), ("goals_against", "ga_ewm")):
        df[name] = grouped[source].transform(
            lambda s: s.shift(1).ewm(
                halflife=STRENGTH_HALFLIFE, min_periods=STRENGTH_MIN_PERIODS
            ).mean()
        )

    df["attack_raw"] = (df["gf_ewm"] / df["league_avg_goals"]).fillna(1.0).clip(0.2, 5.0)
    df["defence_raw"] = (df["ga_ewm"] / df["league_avg_goals"]).fillna(1.0).clip(0.2, 5.0)

    # --- pass 2: adjust for the opponent faced ---------------------------
    # The opponent's raw values come from the same match row, which was built
    # only from *their* matches before this one. No leakage.
    opponent = (
        df[["match_id", "team", "attack_raw", "defence_raw"]]
        .rename(columns={
            "team": "opponent",
            "attack_raw": "opp_attack_raw",
            "defence_raw": "opp_defence_raw",
        })
    )
    df = df.merge(opponent, on=["match_id", "opponent"], how="left")
    df["opp_attack_raw"] = df["opp_attack_raw"].fillna(1.0)
    df["opp_defence_raw"] = df["opp_defence_raw"].fillna(1.0)

    df["attack"] = (
        df["attack_raw"] / df["opp_defence_raw"].clip(0.4, 2.5) ** OPPONENT_ADJUSTMENT
    ).clip(0.2, 5.0)
    df["defence"] = (
        df["defence_raw"] / df["opp_attack_raw"].clip(0.4, 2.5) ** OPPONENT_ADJUSTMENT
    ).clip(0.2, 5.0)

    return df.drop(columns=["gf_ewm", "ga_ewm"])


# --------------------------------------------------------------------------
# Head-to-head
# --------------------------------------------------------------------------


def add_head_to_head(matches: pd.DataFrame, recent_n: int = H2H_RECENT_N) -> pd.DataFrame:
    """Attach head-to-head history, always from the *home* team's perspective.

    For each meeting we store (home_team, home_score, away_score). A pair key is
    order-independent, so a fixture counts as history regardless of who was
    listed at home.

    Produces (for the home side): h2h_played (all-time meetings),
    h2h_gd (mean goal difference in the last `recent_n`), h2h_win and h2h_draw
    (shares over those same last `recent_n`). Away rows are derived by flipping
    the goal difference and complementing the outcome shares -- exact, since
    W/D/L partition the sample.
    """
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    history: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)

    played = np.zeros(len(df), dtype=int)
    gd = np.zeros(len(df), dtype=float)
    win = np.zeros(len(df), dtype=float)
    draw = np.zeros(len(df), dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        key = (row.home_team, row.away_team) if row.home_team <= row.away_team \
            else (row.away_team, row.home_team)
        meetings = history[key]
        played[i] = len(meetings)

        recent = meetings[-recent_n:]
        if recent:
            diffs, wins, draws = [], [], []
            for h, hs, as_ in recent:
                if h == row.home_team:
                    diffs.append(hs - as_)
                else:
                    diffs.append(as_ - hs)
                wins.append(1.0 if diffs[-1] > 0 else 0.0)
                draws.append(1.0 if diffs[-1] == 0 else 0.0)
            gd[i] = float(np.mean(diffs))
            win[i] = float(np.mean(wins))
            draw[i] = float(np.mean(draws))

        # Update AFTER recording, so this meeting is not in its own history.
        history[key].append((row.home_team, int(row.home_score), int(row.away_score)))

    df["h2h_played"] = played
    df["h2h_gd_home"] = gd
    df["h2h_win_home"] = win
    df["h2h_draw"] = draw
    return df


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def compute_features(
    matches: pd.DataFrame,
    *,
    start_rating: float = START_RATING,
    k_base: float = K_BASE,
    home_advantage: float = HOME_ADVANTAGE_ELO,
    form_windows: tuple[int, ...] = FORM_WINDOWS,
) -> pd.DataFrame:
    """Full feature table: one row per team per match, all features pre-match."""
    from .data import to_long_format  # local import avoids a circular import

    if "tournament_category" not in matches.columns:
        matches = matches.assign(
            tournament_category=matches["tournament"].map(config.classify_tournament)
        )

    elo = add_elo_ratings(
        matches, start_rating=start_rating, k_base=k_base, home_advantage=home_advantage
    )
    h2h = add_head_to_head(matches)

    long = to_long_format(matches)

    # Elo and h2h are computed at match level; attach each team's perspective.
    elo_cols = elo[["match_id", "elo_home_pre", "elo_away_pre"]]
    long = long.merge(elo_cols, on="match_id", how="left")
    long["elo_team"] = np.where(long["is_home"] == 1, long["elo_home_pre"], long["elo_away_pre"])
    long["elo_opponent"] = np.where(long["is_home"] == 1, long["elo_away_pre"], long["elo_home_pre"])
    long["elo_diff"] = long["elo_team"] - long["elo_opponent"]

    h2h_cols = h2h[["match_id", "h2h_played", "h2h_gd_home", "h2h_win_home", "h2h_draw"]]
    long = long.merge(h2h_cols, on="match_id", how="left")
    is_home = long["is_home"] == 1
    long["h2h_gd"] = np.where(is_home, long["h2h_gd_home"], -long["h2h_gd_home"])
    long["h2h_win"] = np.where(
        is_home, long["h2h_win_home"], 1.0 - long["h2h_win_home"] - long["h2h_draw"]
    )
    long = long.drop(columns=["h2h_gd_home", "h2h_win_home"])

    long = add_rolling_form(long, windows=form_windows)
    long = add_strength(long, matches)

    # Fill the form NaNs with the league prior: "no history" becomes "average",
    # and `history_matches` keeps that assumption visible to the model.
    form_cols = [c for c in long.columns if c.startswith("form_")]
    long[form_cols] = long[form_cols].astype(float)
    for col in form_cols:
        long[col] = long[col].fillna(long["league_avg_goals"] if "gf" in col or "ga" in col
                                     else 0.0)

    long["h2h_played"] = long["h2h_played"].fillna(0).astype(int)
    for col in ("h2h_gd", "h2h_win"):
        long[col] = long[col].fillna(0.0)

    return long.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]) \
               .reset_index(drop=True)


# Columns the models in steps 4-5 are allowed to consume. Note what is absent:
# `is_home` (at a neutral venue it is arbitrary noise) and the raw score columns.
FEATURE_COLUMNS = [
    "elo_team",
    "elo_opponent",
    "elo_diff",
    "home_advantage",
    "neutral",
    "history_matches",
    "form_gf_5",
    "form_ga_5",
    "form_points_5",
    "form_win_5",
    "form_gf_10",
    "form_ga_10",
    "form_points_10",
    "form_win_10",
    "attack",
    "defence",
    "h2h_played",
    "h2h_gd",
    "h2h_win",
]


def build_feature_table(
    matches: pd.DataFrame | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Compute the full feature table and (optionally) persist it to parquet."""
    config.ensure_dirs()
    if matches is None:
        from .data import load_results

        matches = load_results()

    features = compute_features(matches)
    if write:
        features.to_parquet(config.FEATURES_PARQUET, index=False)
    return features
