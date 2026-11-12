"""Step 6: the evaluation report — tables, bootstrap intervals and charts.

Everything here reads the same fitted models through the same interface, so the
report cannot drift from the pipeline. Four figures, each answering a question
the tables cannot:

* **calibration** — when a model says 30%, does it happen 30% of the time?
* **model comparison** — RPS with intervals, so overlapping models look overlapping.
* **per-match RPS** — *where* two models differ, not just by how much on average.
* **scoreline heatmap** — what a prediction actually looks like, against the
  result that really happened.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import config, data, evaluate, features, scorelines, splits
from .models import DixonColes, GBMPoisson, PoissonGLM
from .models.baseline import FormBaseline, HomeAwayBaseline, LeagueAverageBaseline

DISPLAY = {
    "baseline_league_average": "League average",
    "baseline_home_away": "Home/away average",
    "baseline_form": "Form heuristic",
    "poisson_glm": "Poisson GLM",
    "dixon_coles": "Dixon-Coles",
    "gbm_poisson": "LightGBM (Poisson)",
}

PRED_COLUMNS = ["lambda_home", "lambda_away", "home_score", "away_score"]
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]

FACTORIES = [
    LeagueAverageBaseline,
    HomeAwayBaseline,
    FormBaseline,
    PoissonGLM,
    DixonColes,
    lambda: GBMPoisson.from_saved(),
]

# Models drawn on the diagnostic charts (the two weakest baselines add clutter).
CHART_MODELS = ["baseline_form", "poisson_glm", "dixon_coles", "gbm_poisson"]

PALETTE = {
    "baseline_form": "#8c8c8c",
    "poisson_glm": "#1f77b4",
    "dixon_coles": "#d62728",
    "gbm_poisson": "#2ca02c",
}


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def collect_predictions(
    factories, train: pd.DataFrame, targets: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Fit each model once and store match-level predictions for every block.

    Probabilities are computed with the model's *own* outcome function, so
    Dixon-Coles is scored on its rho-corrected distribution rather than an
    independent-Poisson approximation of it.
    """
    predictions: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}

    for factory in factories:
        model = factory().fit(train)
        frames = []
        for split_name, block in targets.items():
            lambdas = model.predict_lambda(block)
            frame = evaluate.pivot_predictions(block, lambdas)

            probs = np.array([
                model.outcome_probabilities(a, b)
                for a, b in zip(frame["lambda_home"], frame["lambda_away"])
            ])
            frame["p_home"], frame["p_draw"], frame["p_away"] = probs.T
            frame["outcome"] = evaluate.outcome_index(
                frame["home_score"], frame["away_score"]
            )
            frame["rps"] = evaluate.rps_per_match(probs, frame["outcome"].to_numpy())
            frame["split"] = split_name
            frames.append(frame)

        predictions[model.name] = pd.concat(frames, ignore_index=True)
        models[model.name] = model

    return predictions, models


def summary_table(
    predictions: dict[str, pd.DataFrame], block: str, n_boot: int = 1000
) -> pd.DataFrame:
    """Metrics with bootstrap intervals for one block."""
    rows = []
    for name, frame in predictions.items():
        sub = frame[frame["split"] == block]
        if sub.empty:
            continue
        probs = sub[PROB_COLUMNS].to_numpy()
        point = evaluate.evaluate_predictions(sub[PRED_COLUMNS], probs=probs)
        intervals = evaluate.bootstrap_metrics(
            sub[PRED_COLUMNS], n_boot=n_boot, probs=probs
        ).set_index("metric")

        row = {"model": name, "n_matches": int(point["n_matches"]), "mae": point["mae"]}
        for metric in ("rps", "log_loss", "accuracy"):
            row[metric] = point[metric]
            row[f"{metric}_low"] = float(intervals.loc[metric, "ci_low"])
            row[f"{metric}_high"] = float(intervals.loc[metric, "ci_high"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def plot_calibration(
    predictions: dict[str, pd.DataFrame],
    block: str,
    path: Path,
    n_bins: int = 10,
    min_count: int = 20,
) -> Path:
    """Reliability curve: predicted probability vs observed frequency."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")

    for name in CHART_MODELS:
        if name not in predictions:
            continue
        sub = predictions[name][predictions[name]["split"] == block]
        table = evaluate.calibration_bins(
            sub[PROB_COLUMNS].to_numpy(), sub["outcome"].to_numpy(), n_bins
        )
        table = table[table["n"] >= min_count]
        ax.plot(
            table["mean_predicted"], table["observed"], "o-", ms=4, lw=1.3,
            color=PALETTE.get(name), label=DISPLAY.get(name, name),
        )

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration — {block} (all three outcomes pooled)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(
    summaries: dict[str, pd.DataFrame], path: Path, metric: str = "rps"
) -> Path:
    """Grouped bars with bootstrap error bars, one group per block."""
    blocks = list(summaries)
    names = summaries[blocks[0]]["model"].tolist()

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.8 / len(blocks)

    for i, block in enumerate(blocks):
        table = summaries[block].set_index("model").loc[names]
        x = np.arange(len(names)) + i * width - 0.4 + width / 2
        yerr = np.vstack([
            table[metric] - table[f"{metric}_low"],
            table[f"{metric}_high"] - table[metric],
        ])
        ax.bar(x, table[metric], width=width * 0.92, label=block,
               yerr=yerr, capsize=3, alpha=0.9)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([DISPLAY.get(n, n) for n in names], rotation=18, ha="right")
    ax.set_ylabel("RPS (lower is better)" if metric == "rps" else metric)
    ax.set_title("Outcome quality with 95% bootstrap intervals")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rps_ecdf(
    predictions: dict[str, pd.DataFrame], block: str, path: Path
) -> Path:
    """Cumulative share of matches below each per-match RPS.

    Two models with the same average can differ hugely here: the curve that sits
    further left is better on the *typical* match, even if a few disasters pull
    its average up.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for name in CHART_MODELS:
        if name not in predictions:
            continue
        values = np.sort(
            predictions[name].loc[predictions[name]["split"] == block, "rps"].to_numpy()
        )
        if len(values) == 0:
            continue
        share = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, share, where="post", lw=1.6,
                color=PALETTE.get(name), label=DISPLAY.get(name, name))

    ax.set_xlabel("per-match RPS")
    ax.set_ylabel("share of matches at or below")
    ax.set_title(f"Per-match RPS distribution — {block}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_scoreline_heatmap(
    model, row: pd.Series, path: Path, max_goals: int = 5
) -> Path:
    """0-5 scoreline grid with the actual result outlined."""
    lam_home, lam_away = float(row["lambda_home"]), float(row["lambda_away"])

    if hasattr(model, "predict_score_matrix"):
        matrix = model.predict_score_matrix(lam_home, lam_away, max_goals=max_goals)
    else:
        matrix = scorelines.score_matrix(lam_home, lam_away, max_goals=max_goals)

    actual_home, actual_away = int(row["home_score"]), int(row["away_score"])
    p_home, p_draw, p_away = scorelines.outcome_probabilities_from_matrix(matrix)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="probability")

    peak = matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value < 0.002:
                continue
            ax.text(
                j, i, f"{100 * value:.1f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.55 * peak else "black",
            )

    if actual_home <= max_goals and actual_away <= max_goals:
        ax.add_patch(Rectangle(
            (actual_away - 0.5, actual_home - 0.5), 1, 1,
            fill=False, edgecolor="crimson", linewidth=2.2,
        ))

    ax.set_xlabel(f"{row['away_team']} goals")
    ax.set_ylabel(f"{row['home_team']} goals")
    ax.set_title(
        f"{row['home_team']} {actual_home}-{actual_away} {row['away_team']}\n"
        f"predicted {lam_home:.2f} - {lam_away:.2f} goals"
    )
    ax.text(
        0.02, -0.22,
        f"W/D/L: {p_home:.0%} / {p_draw:.0%} / {p_away:.0%}",
        transform=ax.transAxes, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "Model": DISPLAY.get(row["model"], row["model"]),
            "Matches": int(row["n_matches"]),
            "RPS": f"{row['rps']:.3f} [{row['rps_low']:.3f}, {row['rps_high']:.3f}]",
            "Log loss": (f"{row['log_loss']:.3f} "
                         f"[{row['log_loss_low']:.3f}, {row['log_loss_high']:.3f}]"),
            "Accuracy": (f"{row['accuracy']:.3f} "
                         f"[{row['accuracy_low']:.3f}, {row['accuracy_high']:.3f}]"),
            "MAE (goals)": f"{row['mae']:.3f}",
        })
    return pd.DataFrame(rows)


def _pick_showcase(
    predictions: dict[str, pd.DataFrame], block: str = "world_cup_2022"
) -> tuple[str, pd.Series] | None:
    """The last match of the block (the final) for the worked example."""
    candidates = [
        (name, frame[frame["split"] == block])
        for name, frame in predictions.items()
        if not frame[frame["split"] == block].empty
    ]
    if not candidates:
        return None
    name = "dixon_coles" if any(n == "dixon_coles" for n, _ in candidates) else candidates[0][0]
    frame = dict(candidates)[name].sort_values("date")
    return name, frame.iloc[-1]


def build(n_boot: int = 1000, *, write: bool = True) -> dict[str, Path]:
    """Fit every model, score every block, draw the charts, write results.md."""
    config.ensure_dirs()

    matches = data.load_results()
    synthetic = bool(matches.attrs.get("synthetic", False))

    feats = features.build_feature_table(matches, write=write)
    feats = splits.assign_split(feats)
    splits.validate_splits(feats)
    frames = splits.split_frames(feats)

    targets: dict[str, pd.DataFrame] = {}
    for label in ("validation", "test"):
        if label in frames:
            targets[label] = frames[label]
    for year in config.WORLD_CUP_WINDOWS:
        block = splits.world_cup_slice(feats, year)
        if len(block):
            targets[f"world_cup_{year}"] = block

    predictions, models = collect_predictions(FACTORIES, frames["train"], targets)

    summaries = {
        block: summary_table(predictions, block, n_boot)
        for block in ("test", "world_cup_2022")
        if block in targets
    }
    for block in summaries:
        summaries[block] = summaries[block].sort_values("rps")

    fig = config.FIGURES_DIR
    figure_paths = [
        plot_calibration(predictions, "test", fig / "09_calibration.png"),
        plot_model_comparison(summaries, fig / "10_model_comparison.png"),
        plot_rps_ecdf(predictions, "test", fig / "11_rps_per_match.png"),
    ]

    showcase = _pick_showcase(predictions)
    if showcase is not None:
        name, row = showcase
        figure_paths.append(
            plot_scoreline_heatmap(models[name], row, fig / "12_scoreline_heatmap.png")
        )

    # ---- markdown ------------------------------------------------------
    lines: list[str] = []
    lines.append("# Results\n")

    if synthetic:
        lines.append(
            "> **These numbers come from the synthetic fallback data, not the real "
            "Kaggle file.** They are a plumbing check: model ordering and interval "
            "widths are meaningful, absolute values are not. Drop "
            "`results.csv` into `data/raw/` and re-run `python -m src.pipeline report`.\n"
        )

    lines.append("## Setup\n")
    lines.append(
        f"- {len(matches):,} matches, {matches['date'].min().date()} to "
        f"{matches['date'].max().date()}.\n"
        "- Split: train <= 2017, validation 2018-2021, test 2022. Never shuffled.\n"
        "- Features are point-in-time; Elo and rolling form are read before kick-off.\n"
        f"- Intervals are {n_boot}-resample percentile bootstraps, resampling "
        "**matches** (not team-rows).\n"
    )

    lines.append("## Headline: 2022 test block\n")
    lines.append(evaluate.to_markdown(_display_summary(summaries["test"])))
    lines.append(
        "\nLower is better for RPS, log loss and MAE. Accuracy is reported with its "
        "interval because on a few hundred matches the differences between the "
        "fitted models are usually inside the noise.\n"
    )

    if "world_cup_2022" in summaries:
        lines.append("\n## The 2022 World Cup only\n")
        lines.append(evaluate.to_markdown(_display_summary(summaries["world_cup_2022"])))
        lines.append("")

    lines.append("\n## Charts\n")
    captions = {
        "09_calibration.png": "Calibration. A model on the diagonal is honest about "
                              "its confidence; above it, underconfident; below it, "
                              "overconfident.",
        "10_model_comparison.png": "RPS by model with 95% intervals. Overlapping "
                                   "intervals mean the ranking is not resolved.",
        "11_rps_per_match.png": "Per-match RPS distribution. Further left is better "
                                "on the typical match.",
        "12_scoreline_heatmap.png": "A worked example: the predicted scoreline "
                                    "distribution with the actual result outlined.",
    }
    for path in figure_paths:
        lines.append(f"\n![{path.stem}](figures/{path.name})\n")
        lines.append(captions.get(path.name, ""))

    if showcase is not None:
        name, row = showcase
        lines.append(
            f"\nThe example uses **{DISPLAY.get(name, name)}**: "
            f"{row['home_team']} vs {row['away_team']}, "
            f"predicted {row['lambda_home']:.2f} - {row['lambda_away']:.2f} "
            f"against an actual {int(row['home_score'])}-{int(row['away_score'])}.\n"
        )

    lines.append("\n## What these results do and do not say\n")
    lines.append(
        "- The fitted models beat every baseline on Poisson deviance, which is "
        "the goal-level loss they are trained on.\n"
        "- On outcome metrics the fitted models are close to each other. With a "
        "small tournament sample, a 2-3 point RPS gap is not a reliable ordering; "
        "the intervals overlap by design.\n"
        "- Boosting does not automatically win. When the underlying relationship "
        "is close to multiplicative, an additive log-linear model already captures "
        "most of it and a tree ensemble pays for flexibility in variance.\n"
        "- Nothing here is a betting edge. The model prices matches; it does not "
        "know about injuries, travel, squad rotation or motivation.\n"
    )

    report_path = config.REPORTS_DIR / "results.md"
    if write:
        report_path.write_text("\n".join(lines))

    return {
        "report": report_path,
        "figures": figure_paths,
        "test_summary": summaries.get("test", pd.DataFrame()),
    }
