"""Step 7: `predict_match` — from two team names to a full scoreline distribution.

The hard part is not the model, it is building a feature row for a match that has
not happened. Every feature in this project is "state of the world before
kick-off", so for a hypothetical fixture we need each team's **latest known
state**: their most recent Elo rating, their most recent rolling form, and their
most recent attack/defence estimates. That is what `team_state` assembles, and
it is deliberately the last row of each team's point-in-time history rather than
anything computed with hindsight.

The model is loaded from a pickled artifact (`python -m src.pipeline fit`). If no
artifact exists the predictor refits on the fly, so a fresh clone works without
an extra step.
"""

from __future__ import annotations

import difflib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, features, scorelines, splits
from .models import DixonColes, GBMPoisson, PoissonGLM

DEFAULT_MODEL = "poisson_glm"

MODEL_FACTORIES = {
    "poisson_glm": PoissonGLM,
    "dixon_coles": DixonColes,
    "gbm_poisson": lambda: GBMPoisson.from_saved(),
}

# Per-team state carried forward from a team's most recent match.
STATE_COLUMNS = [
    "form_gf_5", "form_ga_5", "form_points_5", "form_win_5",
    "form_gf_10", "form_ga_10", "form_points_10", "form_win_10",
    "attack", "defence", "history_matches",
]


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class MatchPrediction:
    """Everything you want to know about a single predicted fixture."""

    team_a: str
    team_b: str
    neutral: bool
    tournament: str
    model_name: str
    expected_goals_a: float
    expected_goals_b: float
    p_win_a: float
    p_draw: float
    p_win_b: float
    score_matrix: pd.DataFrame          # rows: team_a goals, columns: team_b goals
    top_scorelines: list[tuple[int, int, float]]

    @property
    def most_likely_scoreline(self) -> str:
        home, away, _ = self.top_scorelines[0]
        return f"{home}-{away}"

    def as_dict(self) -> dict:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "neutral": self.neutral,
            "tournament": self.tournament,
            "model": self.model_name,
            "expected_goals_a": round(self.expected_goals_a, 4),
            "expected_goals_b": round(self.expected_goals_b, 4),
            "p_win_a": round(self.p_win_a, 4),
            "p_draw": round(self.p_draw, 4),
            "p_win_b": round(self.p_win_b, 4),
            "most_likely_scoreline": self.most_likely_scoreline,
        }

    def summary(self) -> str:
        venue = "neutral venue" if self.neutral else f"home advantage to {self.team_a}"
        lines = [
            f"{self.team_a} vs {self.team_b}  ({venue}, {self.tournament})",
            f"model: {self.model_name}",
            "",
            f"expected goals : {self.team_a} {self.expected_goals_a:.2f}"
            f"  -  {self.expected_goals_b:.2f} {self.team_b}",
            f"win / draw / win: {self.p_win_a:.1%}  /  {self.p_draw:.1%}"
            f"  /  {self.p_win_b:.1%}",
            f"most likely    : {self.most_likely_scoreline}",
            "",
            "top scorelines:",
        ]
        for home, away, probability in self.top_scorelines:
            lines.append(f"  {home}-{away}   {probability:.1%}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Team state and fixture construction
# --------------------------------------------------------------------------


def team_state(matches: pd.DataFrame, feat_table: pd.DataFrame) -> pd.DataFrame:
    """Latest known state per team: final Elo plus most recent rolling features."""
    elo = features.add_elo_ratings(matches)
    final_elo = (
        pd.concat([
            elo[["date", "home_team", "elo_home_post"]]
            .rename(columns={"home_team": "team", "elo_home_post": "elo"}),
            elo[["date", "away_team", "elo_away_post"]]
            .rename(columns={"away_team": "team", "elo_away_post": "elo"}),
        ])
        .sort_values("date")
        .groupby("team")
        .tail(1)
        .set_index("team")
    )

    latest = (
        feat_table.sort_values(["team", "date", "match_id"])
        .groupby("team")
        .tail(1)
        .set_index("team")
    )

    state = latest[STATE_COLUMNS].join(final_elo["elo"], how="left")
    state["elo"] = state["elo"].fillna(features.START_RATING)
    return state


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str, recent: int = 10) -> dict:
    """Previous meetings, from `team_a`'s point of view."""
    pair = matches[
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b))
        | ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    ].sort_values("date")

    if pair.empty:
        return {"played": 0, "gd": 0.0, "win": 0.0, "draw": 0.0}

    goals_a = np.where(pair["home_team"] == team_a, pair["home_score"], pair["away_score"])
    goals_b = np.where(pair["home_team"] == team_a, pair["away_score"], pair["home_score"])
    diffs = (goals_a - goals_b)[-recent:]

    return {
        "played": int(len(pair)),
        "gd": float(np.mean(diffs)),
        "win": float(np.mean(diffs > 0)),
        "draw": float(np.mean(diffs == 0)),
    }


def fixture_rows(
    state: pd.DataFrame,
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    neutral: bool,
    tournament_category: str,
) -> pd.DataFrame:
    """Two team-rows describing a fixture that has not been played.

    `team_a` is listed first, so it is the nominal home side. When `neutral` is
    true, home advantage is switched off for both rows -- the model reads that
    flag, not the team ordering.
    """
    row_a = state.loc[team_a]
    row_b = state.loc[team_b]
    h2h = head_to_head(matches, team_a, team_b)

    rows = []
    for team, opponent, own, other, is_home in (
        (team_a, team_b, row_a, row_b, 1),
        (team_b, team_a, row_b, row_a, 0),
    ):
        # Head-to-head is stored from team_a's perspective; flip it for team_b.
        if is_home:
            h2h_gd, h2h_win = h2h["gd"], h2h["win"]
        else:
            h2h_gd = -h2h["gd"]
            h2h_win = 1.0 - h2h["win"] - h2h["draw"]

        row = {
            "match_id": -1,
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "neutral": bool(neutral),
            "home_advantage": int(is_home == 1 and not neutral),
            "tournament_category": tournament_category,
            "elo_team": float(own["elo"]),
            "elo_opponent": float(other["elo"]),
            "elo_diff": float(own["elo"] - other["elo"]),
            "h2h_played": h2h["played"],
            "h2h_gd": h2h_gd,
            "h2h_win": h2h_win,
        }
        for column in STATE_COLUMNS:
            row[column] = float(own[column])
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def artifact_path(model_name: str) -> Path:
    return config.MODELS_DIR / f"{model_name}.pkl"


def save_artifact(model, metadata: dict) -> Path:
    """Pickle a fitted model plus the provenance of its training data."""
    path = artifact_path(model.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    return path


def load_artifact(model_name: str) -> tuple[object, dict]:
    with open(artifact_path(model_name), "rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload["metadata"]


# --------------------------------------------------------------------------
# Team-name resolution
# --------------------------------------------------------------------------


def resolve_team(name: str, teams: list[str]) -> str:
    """Map a user-supplied name onto a team in the data, or fail helpfully."""
    if name in teams:
        return name

    candidates = difflib.get_close_matches(name, teams, n=3, cutoff=0.6)
    if candidates:
        # Accept the best match only when it is clearly better than the runner-up
        # ("Argentia" -> Argentina, not a coin flip between Argentina and Algeria).
        best = candidates[0]
        similarity = difflib.SequenceMatcher(None, name.lower(), best.lower()).ratio()
        if similarity >= 0.8:
            return best

    suggestion = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
    raise KeyError(f"Unknown team {name!r}.{suggestion}")


# --------------------------------------------------------------------------
# The predictor
# --------------------------------------------------------------------------


class Predictor:
    """A fitted model plus the team state needed to build future fixtures."""

    def __init__(self, model, matches: pd.DataFrame, state: pd.DataFrame,
                 model_name: str, metadata: dict | None = None):
        self.model = model
        self.matches = matches
        self.state = state
        self.model_name = model_name
        self.metadata = metadata or {}

    # --- construction -------------------------------------------------
    @classmethod
    def from_data(cls, model_name: str = DEFAULT_MODEL,
                  matches: pd.DataFrame | None = None) -> "Predictor":
        """Fit from scratch on everything before the test block (train+validation)."""
        matches = matches if matches is not None else data.load_results()
        feats = features.build_feature_table(matches, write=False)
        feats = splits.assign_split(feats)

        train = feats[feats["split"].isin(["train", "validation"])]
        model = MODEL_FACTORIES[model_name]().fit(train)
        return cls(
            model=model,
            matches=matches,
            state=team_state(matches, feats),
            model_name=model_name,
            metadata={"trained_on": ["train", "validation"], "n_team_rows": int(len(train))},
        )

    @classmethod
    def from_saved(cls, model_name: str = DEFAULT_MODEL,
                   matches: pd.DataFrame | None = None) -> "Predictor":
        """Load a saved artifact, falling back to fitting if none exists."""
        path = artifact_path(model_name)
        if not path.exists():
            predictor = cls.from_data(model_name, matches=matches)
            print(f"[predict] no artifact for {model_name}; fitted on the fly")
            return predictor

        model, metadata = load_artifact(model_name)
        matches = matches if matches is not None else data.load_results()
        feats = features.build_feature_table(matches, write=False)
        return cls(
            model=model,
            matches=matches,
            state=team_state(matches, feats),
            model_name=model_name,
            metadata=metadata,
        )

    # --- prediction ---------------------------------------------------
    @property
    def teams(self) -> list[str]:
        return sorted(self.state.index)

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        max_goals: int = 5,
    ) -> MatchPrediction:
        """Expected goals, scoreline matrix and W/D/L for one fixture.

        Parameters
        ----------
        team_a, team_b
            Team names as they appear in the dataset. Near-misses are corrected
            automatically (one close match) or reported with suggestions.
        neutral
            True for a neutral venue (World Cup), False to give `team_a` home
            advantage.
        tournament
            Either a tournament category ("world_cup") or free text
            ("FIFA World Cup"), classified the same way the training data was.
        """
        a = resolve_team(team_a, self.teams)
        b = resolve_team(team_b, self.teams)
        if a == b:
            raise ValueError("a team cannot play itself")

        category = (
            tournament if tournament in config.TOURNAMENT_CATEGORIES
            else config.classify_tournament(tournament)
        )

        rows = fixture_rows(
            self.state, self.matches, a, b,
            neutral=neutral, tournament_category=category,
        )
        lambdas = self.model.predict_lambda(rows)
        lam_a, lam_b = float(lambdas[0]), float(lambdas[1])

        # Outcome probabilities come from the model's full distribution, not the
        # 0-5 display grid (which throws away the tail).
        p_a, p_draw, p_b = self.model.outcome_probabilities(lam_a, lam_b)

        if hasattr(self.model, "predict_score_matrix"):
            matrix = self.model.predict_score_matrix(lam_a, lam_b, max_goals=max_goals)
        else:
            matrix = scorelines.score_matrix(lam_a, lam_b, max_goals=max_goals)

        top = scorelines.most_likely_scorelines(lam_a, lam_b, top_n=5)

        return MatchPrediction(
            team_a=a,
            team_b=b,
            neutral=bool(neutral),
            tournament=tournament,
            model_name=self.model_name,
            expected_goals_a=lam_a,
            expected_goals_b=lam_b,
            p_win_a=p_a,
            p_draw=p_draw,
            p_win_b=p_b,
            score_matrix=pd.DataFrame(
                100 * matrix,   # full precision: rounding here breaks the sum
                index=pd.Index(range(matrix.shape[0]), name=f"{a} goals"),
                columns=pd.Index(range(matrix.shape[1]), name=f"{b} goals"),
            ),
            top_scorelines=top,
        )


# --------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------

_CACHE: dict[str, Predictor] = {}


def get_predictor(model_name: str = DEFAULT_MODEL) -> Predictor:
    """Load (and cache) a predictor for interactive use."""
    if model_name not in _CACHE:
        _CACHE[model_name] = Predictor.from_saved(model_name)
    return _CACHE[model_name]


def predict_match(
    team_a: str,
    team_b: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    model_name: str | None = None,
) -> MatchPrediction:
    """Predict a fixture: expected goals, scoreline matrix, win/draw/loss."""
    return get_predictor(model_name or DEFAULT_MODEL).predict_match(
        team_a, team_b, neutral=neutral, tournament=tournament
    )
