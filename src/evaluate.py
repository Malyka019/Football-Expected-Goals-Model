"""Metrics for goal predictions and match outcomes (step 3).

Two levels of prediction, two families of metric
-----------------------------------------------
**Goal level** (one value per team per match): MAE, RMSE, Poisson deviance.
These measure how well the rates match reality, but they are *noise-dominated*:
a team expected to score 1.4 goals scoring 2 is not an error of 0.6, it is a
single draw from a Poisson(1.4). Do not expect MAE below ~0.95 for any model --
that is the inherent noise floor for football goals, and a model that "beats" it
is usually leaking.

**Outcome level** (one value per match): accuracy, log loss, RPS.
- *Accuracy* is the metric everyone quotes and the worst one to optimise: it
  throws away confidence, so a model that says 51% and a model that says 95%
  score identically.
- *Log loss* punishes confident mistakes without bound. It is the right
  scoring rule for a probability forecast, but it is hard to interpret.
- *RPS* (ranked probability score) is the football-specific favourite: it is a
  proper scoring rule that respects the natural ordering home win < draw <
  away win, so predicting a home win when the away side wins costs more than
  predicting a draw does. Lower is better; 0 is perfect.

Uncertainty
-----------
A World Cup is 64 matches. Any accuracy difference under ~5 percentage points
between two models on that sample is noise, so every metric here comes with a
bootstrap interval. Resampling is done **per match**, not per team-row: the two
rows of a match are the same event, and treating them as independent understates
the interval by roughly a factor of root two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, scorelines

EPS = 1e-15

GOAL_METRICS = ["mae", "rmse", "poisson_deviance", "mean_predicted_goals"]
OUTCOME_METRICS = ["accuracy", "log_loss", "rps"]
METRIC_KEYS = GOAL_METRICS + OUTCOME_METRICS

# Human-readable labels, used in the results tables and the README.
METRIC_LABELS = {
    "mae": "MAE (goals)",
    "rmse": "RMSE (goals)",
    "poisson_deviance": "Poisson deviance",
    "mean_predicted_goals": "mean predicted goals",
    "accuracy": "W/D/L accuracy",
    "log_loss": "log loss",
    "rps": "RPS",
}


# --------------------------------------------------------------------------
# Goal-level metrics
# --------------------------------------------------------------------------


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(diff ** 2)))


def poisson_deviance(y_true: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance: 2 * [y log(y/mu) - (y - mu)].

    The natural loss for a Poisson model -- it is the quantity a Poisson GLM
    minimises, so comparing models on it is comparing them on their own terms.
    The 0 * log(0) = 0 convention is handled by the `where`.
    """
    y = np.asarray(y_true, float)
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    term = np.where(y == 0, mu, y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu))
    return float(np.mean(2.0 * term))


# --------------------------------------------------------------------------
# Outcome-level metrics
# --------------------------------------------------------------------------


def outcome_index(home_score, away_score) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home = np.asarray(home_score, float)
    away = np.asarray(away_score, float)
    return np.where(home > away, 0, np.where(home == away, 1, 2))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Share of matches where the most likely outcome was the actual one."""
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(outcomes)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """-mean(log p(actual outcome)), clipped away from zero."""
    p = np.clip(np.asarray(probs, float), EPS, 1.0)
    idx = np.arange(len(p))
    return float(-np.mean(np.log(p[idx, np.asarray(outcomes)])))


def rps_per_match(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """RPS for each match separately (the mean of this is the overall RPS).

        RPS = 1/(K-1) * sum_{i=1}^{K-1} (CDF_pred(i) - CDF_obs(i))^2

    with K = 3 ordered outcomes. Per-match values are what the diagnostic charts
    use: two models can have the same average RPS and be wrong on very different
    matches, and only the per-match view shows that.
    """
    p = np.asarray(probs, float)
    o = np.asarray(outcomes)
    pred_cdf = np.stack([p[:, 0], p[:, 0] + p[:, 1]], axis=1)
    obs_cdf = np.stack([(o == 0), (o <= 1)], axis=1).astype(float)
    return np.sum((pred_cdf - obs_cdf) ** 2, axis=1) / (3 - 1)


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS for three ordered outcomes, averaged over matches.

    Lower is better; 0 is a certain, correct forecast. A no-information forecast
    scores around 0.22-0.24 on real football, so treat that as the reference for
    "knows nothing".
    """
    return float(np.mean(rps_per_match(probs, outcomes)))


def calibration_bins(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability table: predicted probability vs observed frequency.

    All three outcomes are pooled (each match contributes three (p, hit) pairs),
    which is what you want for a single calibration curve: it answers "when this
    model says 30%, how often did that happen?" without caring which outcome.
    """
    p = np.asarray(probs, float)
    o = np.asarray(outcomes)
    hits = np.stack([o == 0, o == 1, o == 2], axis=1).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p.ravel(), edges) - 1, 0, n_bins - 1)

    frame = pd.DataFrame({"bin": bin_idx, "p": p.ravel(), "hit": hits.ravel()})
    grouped = frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed=("hit", "mean"), n=("p", "size")
    ).reset_index()
    grouped["bin_low"] = edges[grouped["bin"]]
    grouped["bin_high"] = edges[grouped["bin"] + 1]
    return grouped


# --------------------------------------------------------------------------
# Preparing and scoring predictions
# --------------------------------------------------------------------------


@dataclass
class PreparedPredictions:
    """Arrays needed by every metric, computed once."""

    probs: np.ndarray       # (n_matches, 3) home/draw/away
    outcomes: np.ndarray    # (n_matches,)
    goals: np.ndarray       # (2 * n_matches,) actual goals, home block then away
    rates: np.ndarray       # (2 * n_matches,) predicted rates, same order
    n_matches: int


def prepare_predictions(
    preds: pd.DataFrame,
    max_goals: int = scorelines.MAX_GOALS,
    probs_fn=None,
    probs: np.ndarray | None = None,
) -> PreparedPredictions:
    """Turn a match-level prediction table into metric-ready arrays.

    `probs_fn(lam_home, lam_away, max_goals=...)` lets a model supply its own
    scoreline distribution -- Dixon-Coles uses it to score its rho-corrected
    probabilities. When omitted, independent Poisson is used.
    """
    required = {"lambda_home", "lambda_away", "home_score", "away_score"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")

    lam_home = preds["lambda_home"].to_numpy(float)
    lam_away = preds["lambda_away"].to_numpy(float)
    home_score = preds["home_score"].to_numpy(float)
    away_score = preds["away_score"].to_numpy(float)

    if probs is not None:
        # Pre-computed (n, 3) probabilities, e.g. a model's rho-corrected ones.
        probs_array = np.asarray(probs, dtype=float)
        if probs_array.shape != (len(preds), 3):
            raise ValueError(
                f"probs must have shape ({len(preds)}, 3), got {probs_array.shape}"
            )
    elif probs_fn is None:
        probs_array = np.array([
            scorelines.outcome_probabilities(a, b, max_goals=max_goals)
            for a, b in zip(lam_home, lam_away)
        ])
    else:
        probs_array = np.array([
            probs_fn(a, b, max_goals=max_goals)
            for a, b in zip(lam_home, lam_away)
        ])
    return PreparedPredictions(
        probs=probs_array,
        outcomes=outcome_index(home_score, away_score),
        goals=np.concatenate([home_score, away_score]),
        rates=np.concatenate([lam_home, lam_away]),
        n_matches=len(preds),
    )


def metrics_from_prepared(p: PreparedPredictions) -> dict[str, float]:
    """All metrics for already-prepared arrays (used by the bootstrap too)."""
    return {
        "mae": mean_absolute_error(p.goals, p.rates),
        "rmse": root_mean_squared_error(p.goals, p.rates),
        "poisson_deviance": poisson_deviance(p.goals, p.rates),
        "mean_predicted_goals": float(np.mean(p.rates)),
        "accuracy": accuracy(p.probs, p.outcomes),
        "log_loss": log_loss(p.probs, p.outcomes),
        "rps": ranked_probability_score(p.probs, p.outcomes),
    }


def evaluate_predictions(
    preds: pd.DataFrame,
    max_goals: int = scorelines.MAX_GOALS,
    probs_fn=None,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    """Point estimates of every metric for a match-level prediction table."""
    metrics = metrics_from_prepared(
        prepare_predictions(preds, max_goals, probs_fn, probs)
    )
    metrics["n_matches"] = int(len(preds))
    metrics["n_team_rows"] = int(2 * len(preds))
    return metrics


def bootstrap_metrics(
    preds: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = config.RANDOM_SEED,
    max_goals: int = scorelines.MAX_GOALS,
    confidence: float = 0.95,
    probs_fn=None,
    probs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Percentile bootstrap intervals, resampling **matches** with replacement.

    Returns a tidy table: metric, value, ci_low, ci_high. With 64 matches the
    intervals are wide and overlapping between models -- that is the honest
    answer, and it is far more useful than a fourth decimal place.
    """
    p = prepare_predictions(preds, max_goals, probs_fn, probs)
    n = p.n_matches
    if n_boot <= 0 or n == 0:
        point = metrics_from_prepared(p)
        return pd.DataFrame({"metric": list(point), "value": list(point.values())})

    rng = np.random.default_rng(seed)
    draws = {key: np.empty(n_boot) for key in METRIC_KEYS}

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = PreparedPredictions(
            probs=p.probs[idx],
            outcomes=p.outcomes[idx],
            goals=np.concatenate([p.goals[:n][idx], p.goals[n:][idx]]),
            rates=np.concatenate([p.rates[:n][idx], p.rates[n:][idx]]),
            n_matches=n,
        )
        metrics = metrics_from_prepared(sample)
        for key in METRIC_KEYS:
            draws[key][b] = metrics[key]

    alpha = (1.0 - confidence) / 2.0
    point = metrics_from_prepared(p)
    return pd.DataFrame({
        "metric": METRIC_KEYS,
        "value": [point[key] for key in METRIC_KEYS],
        "ci_low": [float(np.quantile(draws[key], alpha)) for key in METRIC_KEYS],
        "ci_high": [float(np.quantile(draws[key], 1.0 - alpha)) for key in METRIC_KEYS],
    })


# --------------------------------------------------------------------------
# Plumbing between the long table and match-level evaluation
# --------------------------------------------------------------------------


def pivot_predictions(long: pd.DataFrame, lambdas: np.ndarray) -> pd.DataFrame:
    """Fold team-level predicted rates into one row per match.

    The long table has two rows per match; evaluation needs one. `goals_for` on
    the home row is the home score, so the pivot is also a check that the
    long-format invariant still holds.
    """
    df = long.assign(predicted_goals=np.asarray(lambdas, dtype=float))

    home = (
        df[df["is_home"] == 1]
        .loc[:, ["match_id", "date", "tournament_category", "neutral",
                 "home_team", "away_team", "goals_for", "predicted_goals"]]
        .rename(columns={"goals_for": "home_score", "predicted_goals": "lambda_home"})
    )
    away = (
        df[df["is_home"] == 0]
        .loc[:, ["match_id", "goals_for", "predicted_goals"]]
        .rename(columns={"goals_for": "away_score", "predicted_goals": "lambda_away"})
    )

    preds = home.merge(away, on="match_id", how="inner", validate="one_to_one")
    return preds.sort_values(["date", "match_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Results tables
# --------------------------------------------------------------------------


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble model/split rows into a rounded comparison table."""
    table = pd.DataFrame(rows)
    ordered = ["model", "split", "n_matches"] + METRIC_KEYS
    ordered = [c for c in ordered if c in table.columns]
    return table[ordered].round(4)


def to_markdown(table: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal markdown table renderer (avoids a `tabulate` dependency)."""
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in table.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_markdown(
    table: pd.DataFrame, path: str | Path, *, title: str, note: str = ""
) -> Path:
    """Write a results table plus an optional note to a markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n"
    if note:
        body += note.strip() + "\n\n"
    body += to_markdown(table) + "\n"
    path.write_text(body)
    return path
