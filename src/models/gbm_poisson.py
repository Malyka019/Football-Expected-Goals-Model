"""Gradient boosting with a Poisson objective (step 5).

Why boosting belongs in this comparison
--------------------------------------
The GLM is additive on the log-rate scale: each feature shifts the expected
goals by a fixed multiplicative amount, everywhere. Football is not that tidy.
Being 200 Elo points stronger might matter more for a team with a settled attack
than for one in transition; home advantage might be worth more to a side that
defends well than to one that does not. A boosted tree ensemble can find those
interactions without being told they exist.

It also comes with the failure mode this project keeps guarding against: with
~10k training rows and a 64-match test set, a flexible model can memorise noise
and look brilliant on validation. Hence the rules used here:

* only the `poisson` objective (so the model still predicts a rate, and the
  evaluation stays comparable);
* shallow, heavily regularised trees (`num_leaves` <= 31, `min_child_samples`
  >= 20, L2 penalty);
* hyperparameters chosen on the **validation** block only, never test;
* no early stopping on test data -- that would be tuning on the thing we are
  supposed to be measuring.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config, evaluate, features
from .base import GoalModel

# Deliberately conservative: this is a small, noisy dataset where a big forest
# wins the leaderboard and loses the World Cup.
DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 1,          # deterministic and fast enough at this size
    "verbose": -1,
}

# Six configurations: 3 x 2. Small on purpose -- a big grid on 700 validation
# matches would just be fitting the noise in the validation set.
TUNING_GRID = [
    {"num_leaves": leaves, "min_child_samples": mcs}
    for leaves in (7, 15, 31)
    for mcs in (20, 60)
]

PARAMS_PATH = config.MODELS_DIR / "gbm_params.json"


def build_design(rows: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix: the numeric features plus tournament as a category.

    `tournament_category` is passed to LightGBM as a pandas Categorical with an
    explicit category list, so training and prediction always agree on the
    encoding even if a split happens to miss a category.
    """
    design = rows[features.FEATURE_COLUMNS].astype(float).copy()
    design["tournament_category"] = pd.Categorical(
        rows["tournament_category"].astype(str),
        categories=list(config.TOURNAMENT_CATEGORIES),
    )
    return design


class GBMPoisson(GoalModel):
    """LightGBM regression with a Poisson objective and log link."""

    name = "gbm_poisson"

    def __init__(self, params: dict | None = None):
        super().__init__()
        self.params = dict(params or {})

    def fit(self, rows: pd.DataFrame) -> "GBMPoisson":
        design = build_design(rows)
        self.feature_names_ = list(design.columns)

        params = {
            **DEFAULT_PARAMS,
            **self.params,
            "objective": "poisson",
            "random_state": config.RANDOM_SEED,
        }
        self.model_ = lgb.LGBMRegressor(**params)
        self.model_.fit(
            design, rows["goals_for"].to_numpy(float),
            categorical_feature=["tournament_category"],
        )
        self.is_fitted = True
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        design = build_design(rows)[self.feature_names_]
        mu = self.model_.predict(design)
        return self._finalise_rates(np.clip(np.asarray(mu, dtype=float), 0.01, 20.0))

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, normalised to percentages."""
        self._check_fitted()
        gains = pd.Series(
            self.model_.feature_importances_.astype(float),
            index=self.model_.feature_name_,
            dtype=float,
        )
        total = gains.sum()
        return (100.0 * gains / total if total > 0 else gains).sort_values(ascending=False)

    @classmethod
    def from_saved(cls, path: str | Path | None = None) -> "GBMPoisson":
        """Load the tuned hyperparameters written by the `tune` command."""
        path = Path(path) if path else PARAMS_PATH
        if path.exists():
            payload = json.loads(path.read_text())
            return cls(params=payload.get("params", {}))
        return cls()


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------


def tune_gbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    grid: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Grid-search hyperparameters on the validation block.

    Selection is by RPS, which is the metric this project actually trusts for
    outcome quality. Returns (results table, best params dict).
    """
    grid = list(grid or TUNING_GRID)
    records = []
    for overrides in grid:
        model = GBMPoisson(params=overrides).fit(train)
        preds = evaluate.pivot_predictions(validation, model.predict_lambda(validation))
        metrics = evaluate.evaluate_predictions(preds)
        records.append({
            **overrides,
            **{key: metrics[key] for key in
               ("mae", "poisson_deviance", "accuracy", "log_loss", "rps")},
        })

    table = pd.DataFrame(records).sort_values("rps").reset_index(drop=True)
    best = {key: table.loc[0, key] for key in grid[0]}
    best = {
        key: int(value) if float(value).is_integer() else float(value)
        for key, value in best.items()
    }
    return table, best


def save_params(params: dict, path: str | Path | None = None, **extra) -> Path:
    """Persist tuned hyperparameters plus provenance for later commands."""
    path = Path(path) if path else PARAMS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, **extra}, indent=2))
    return path
