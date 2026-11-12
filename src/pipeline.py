"""Command-line entry point.

Usage
-----
    python -m src.pipeline build       # clean raw results, write parquet tables
    python -m src.pipeline build --synthetic
    python -m src.pipeline features    # point-in-time feature table
    python -m src.pipeline baseline    # baselines only
    python -m src.pipeline models      # baselines + Poisson GLM + Dixon-Coles
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from datetime import datetime, timezone

from . import config, data, evaluate, features, predict, report, simulate, splits
from .models import DixonColes, GBMPoisson, PoissonGLM
from .models.baseline import FormBaseline, HomeAwayBaseline, LeagueAverageBaseline
from .models.gbm_poisson import save_params, tune_gbm


# --------------------------------------------------------------------------
# build / features
# --------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> None:
    """Build the cleaned match table and the long-format modelling table."""
    matches, long = data.build_datasets(
        synthetic=args.synthetic,
        force=args.force,
        rename_former=args.rename_former,
    )

    report = matches.attrs.get("cleaning_report", {})
    print("\n=== Data build summary ===")
    print(f"matches            : {len(matches):,}")
    print(f"team-match rows    : {len(long):,}  (expected 2 x matches)")
    print(f"date range         : {report.get('date_min')} -> {report.get('date_max')}")
    print(f"teams              : {report.get('n_teams'):,}")
    print(f"neutral-venue games: {report.get('n_neutral'):,}")
    print(f"goals per match    : {(matches['home_score'] + matches['away_score']).mean():.2f}")
    print(f"home win / draw    : {(matches['home_score'] > matches['away_score']).mean():.1%}"
          f" / {(matches['home_score'] == matches['away_score']).mean():.1%}")
    print("\ntournament categories:")
    for category, count in matches["tournament_category"].value_counts().items():
        print(f"  {category:<20} {count:>7,}")
    print(f"\nwrote: {config.MATCHES_CLEAN_PARQUET.relative_to(config.PROJECT_ROOT)}")
    print(f"wrote: {config.MATCHES_LONG_PARQUET.relative_to(config.PROJECT_ROOT)}")
    print("wrote: data/interim/cleaning_report.json")


def cmd_features(args: argparse.Namespace) -> None:
    """Compute the point-in-time feature table and report what came out."""
    matches = data.load_results()
    feats = features.build_feature_table(matches, write=True)

    print("\n=== Feature table ===")
    print(f"team-match rows : {len(feats):,}")
    print(f"matches         : {feats['match_id'].nunique():,}")
    print(f"date range      : {feats['date'].min().date()} -> {feats['date'].max().date()}")
    print(f"columns         : {len(feats.columns)} "
          f"({len(features.FEATURE_COLUMNS)} model features)")

    nulls = feats[features.FEATURE_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    print(f"\nnulls in model features: {'none' if nulls.empty else nulls.to_dict()}")

    points = np.where(feats["result"] == "W", 3, np.where(feats["result"] == "D", 1, 0))
    print(f"corr(elo_diff, points)  : {np.corrcoef(feats['elo_diff'], points)[0, 1]:.3f}")
    print(f"corr(attack, goals_for) : {np.corrcoef(feats['attack'], feats['goals_for'])[0, 1]:.3f}")

    elo = features.add_elo_ratings(matches)
    latest = pd.concat([
        elo[["date", "home_team", "elo_home_post"]]
        .rename(columns={"home_team": "team", "elo_home_post": "elo"}),
        elo[["date", "away_team", "elo_away_post"]]
        .rename(columns={"away_team": "team", "elo_away_post": "elo"}),
    ]).sort_values("date").groupby("team").tail(1)

    top = latest.nlargest(12, "elo")[["team", "elo", "date"]].reset_index(drop=True)
    top["date"] = top["date"].dt.date
    print("\ntop 12 by current Elo rating:")
    print(top.to_string(index=False))

    print(f"\nwrote: {config.FEATURES_PARQUET.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------
# Shared model-comparison plumbing
# --------------------------------------------------------------------------


def _targets(feats: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validation, test, and the two World Cup windows as separate blocks.

    The tournament windows matter: 64 matches hidden inside a year of qualifiers
    would let a model look good on fixtures nobody cares about.
    """
    targets: dict[str, pd.DataFrame] = {}
    for label in ("validation", "test"):
        if label in frames:
            targets[label] = frames[label]
    for year in config.WORLD_CUP_WINDOWS:
        block = splits.world_cup_slice(feats, year)
        if len(block):
            targets[f"world_cup_{year}"] = block
    return targets


def _comparison_rows(factories, train: pd.DataFrame, targets: dict[str, pd.DataFrame]) -> list[dict]:
    """Fit each model on train, score it on every target block."""
    rows: list[dict] = []
    for split_name, block in targets.items():
        for factory in factories:
            model = factory().fit(train)
            preds = evaluate.pivot_predictions(block, model.predict_lambda(block))
            rows.append({
                "model": model.name,
                "split": split_name,
                **evaluate.evaluate_predictions(
                    preds, probs_fn=model.outcome_probabilities
                ),
            })
    return rows


def _run_comparison(
    args: argparse.Namespace,
    factories: list,
    *,
    title: str,
    stem: str,
    note: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """The whole comparison: load, split, fit, score, persist, bootstrap."""
    matches = data.load_results()
    if matches.attrs.get("synthetic") and not args.quiet:
        print("\n!! SYNTHETIC data in use: these numbers check the pipeline, "
              "not football")

    feats = features.build_feature_table(matches, write=True)
    feats = splits.assign_split(feats)
    splits.validate_splits(feats)
    frames = splits.split_frames(feats)
    targets = _targets(feats, frames)
    train = frames["train"]

    print("\n=== Splits ===")
    print(splits.describe_splits(feats).to_string(index=False))

    table = evaluate.results_table(_comparison_rows(factories, train, targets))
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))

    csv_path = config.REPORTS_DIR / f"{stem}.csv"
    table.to_csv(csv_path, index=False)
    md_path = evaluate.write_results_markdown(
        table, config.REPORTS_DIR / f"{stem}.md", title=title, note=note
    )
    print(f"\nwrote {csv_path.relative_to(config.PROJECT_ROOT)}")
    print(f"wrote {md_path.relative_to(config.PROJECT_ROOT)}")

    if args.n_boot > 0 and "test" in frames:
        print(f"\n=== Bootstrap {args.n_boot} resamples, test split ===")
        for factory in factories:
            model = factory().fit(train)
            preds = evaluate.pivot_predictions(
                frames["test"], model.predict_lambda(frames["test"])
            )
            intervals = evaluate.bootstrap_metrics(
                preds, n_boot=args.n_boot, probs_fn=model.outcome_probabilities
            )
            print(f"\n{model.name}")
            print(intervals.round(4).to_string(index=False))

    return table, train, frames


def cmd_baseline(args: argparse.Namespace) -> None:
    """Fit the baselines and score them."""
    _run_comparison(
        args,
        [LeagueAverageBaseline, HomeAwayBaseline, FormBaseline],
        title="Step 3 - baseline results",
        stem="results_baseline",
        note=(
            "Baselines only: no fitted team strengths. `world_cup_2018` and "
            "`world_cup_2022` are the tournament windows themselves; `validation` "
            "and `test` are the full calendar blocks they sit inside."
        ),
    )


def cmd_models(args: argparse.Namespace) -> None:
    """Fit every model so far and compare, then show the fitted parameters."""
    table, train, _ = _run_comparison(
        args,
        [LeagueAverageBaseline, HomeAwayBaseline, FormBaseline, PoissonGLM,
         DixonColes, lambda: GBMPoisson.from_saved()],
        title="Step 4 - model comparison",
        stem="results_models",
        note=(
            "Poisson GLM uses the engineered features; Dixon-Coles fits per-team "
            "attack/defence with two-year time decay and the rho low-score "
            "correction. All models are fitted on the train block only."
        ),
    )

    glm = PoissonGLM().fit(train)
    print("\n=== Poisson GLM coefficients (log goal-rate scale) ===")
    print(glm.params_.round(4).to_string())

    dc = DixonColes().fit(train)
    print(f"\n=== Dixon-Coles ({dc.n_teams_} teams, {dc.n_matches_:,} matches) ===")
    print(f"rho            : {dc.rho_:+.4f}")
    print(f"home advantage : {dc.home_gamma_:.4f}  (x{np.exp(dc.home_gamma_):.2f} goals)")
    print(f"halflife       : {dc.halflife_days_:.0f} days")
    print(f"converged      : {dc.converged_}")

    teams = dc.team_table()
    print("\nstrongest attack parameters:")
    print(teams.sort_values("attack", ascending=False).head(8).round(3).to_string())
    print("\nweakest defences (highest = leaks goals):")
    print(teams.sort_values("defence", ascending=False).head(8).round(3).to_string())

    gbm = GBMPoisson.from_saved().fit(train)
    print("\n=== Gradient boosting feature importance (% of gain) ===")
    importance = gbm.feature_importance()
    print(importance.head(12).round(1).to_string())
    importance_path = config.REPORTS_DIR / "gbm_feature_importance.csv"
    importance.rename("importance_pct").to_csv(importance_path)
    print(f"\nwrote {importance_path.relative_to(config.PROJECT_ROOT)}")

    print(f"\nbest model on test RPS: "
          f"{table.loc[table['split'] == 'test'].sort_values('rps').iloc[0]['model']}")


def cmd_tune(args: argparse.Namespace) -> None:
    """Grid-search gradient-boosting hyperparameters on validation only."""
    matches = data.load_results()
    feats = features.build_feature_table(matches, write=True)
    feats = splits.assign_split(feats)
    splits.validate_splits(feats)
    frames = splits.split_frames(feats)

    print("\n=== GBM tuning (fit on train, scored on validation) ===")
    table, best = tune_gbm(frames["train"], frames["validation"])
    print(table.round(4).to_string(index=False))

    csv_path = config.REPORTS_DIR / "gbm_tuning.csv"
    table.to_csv(csv_path, index=False)
    params_path = save_params(best, tuned_on="validation", selection="lowest RPS")

    print(f"\nbest params: {best}")
    print(f"wrote {params_path.relative_to(config.PROJECT_ROOT)}")
    print(f"wrote {csv_path.relative_to(config.PROJECT_ROOT)}")


def cmd_report(args: argparse.Namespace) -> None:
    """Build reports/results.md plus the calibration, comparison and RPS charts."""
    result = report.build(n_boot=args.n_boot)

    print("\n=== Report ===")
    print(f"wrote {result['report'].relative_to(config.PROJECT_ROOT)}")
    for path in result["figures"]:
        print(f"wrote {path.relative_to(config.PROJECT_ROOT)}")

    summary = result["test_summary"]
    if not summary.empty:
        print("\ntest block, sorted by RPS:")
        print(summary[["model", "rps", "log_loss", "accuracy", "mae"]]
              .round(4).to_string(index=False))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit and pickle the models that `predict_match` and the app load."""
    matches = data.load_results()
    feats = features.build_feature_table(matches, write=True)
    feats = splits.assign_split(feats)

    labels = ["train"] if args.train_only else ["train", "validation"]
    train = feats[feats["split"].isin(labels)]
    print(f"\nfitting on {len(train):,} team-rows "
          f"({train['match_id'].nunique():,} matches, {', '.join(labels)})")
    if matches.attrs.get("synthetic"):
        print("!! SYNTHETIC data: the artifact is a plumbing check, not a predictor")

    for name, factory in predict.MODEL_FACTORIES.items():
        model = factory().fit(train)
        metadata = {
            "model": name,
            "trained_on": labels,
            "n_team_rows": int(len(train)),
            "n_matches": int(train["match_id"].nunique()),
            "last_training_date": str(train["date"].max().date()),
            "synthetic_data": bool(matches.attrs.get("synthetic", False)),
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = predict.save_artifact(model, metadata)
        print(f"  fitted {name:<14} -> {path.relative_to(config.PROJECT_ROOT)}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict one fixture from the command line."""
    result = predict.predict_match(
        args.team_a, args.team_b,
        neutral=args.neutral,
        tournament=args.tournament,
        model_name=args.model,
    )
    print()
    print(result.summary())
    print(f"\nscoreline matrix (%, rows = {result.team_a}, columns = {result.team_b})")
    print(result.score_matrix.round(1).to_string())


def cmd_simulate(args: argparse.Namespace) -> None:
    """Monte Carlo a 32-team World Cup and tabulate champion probabilities."""
    predictor = predict.get_predictor(args.model)
    teams = [t.strip() for t in args.teams.split(",")] if args.teams else None

    result = simulate.run_simulation(
        predictor, teams=teams, n_sims=args.n_sims, seed=args.seed
    )

    print(f"\n=== World Cup simulation ({result.n_sims:,} runs, seed {args.seed}) ===")
    print("\ngroups:")
    for index, group in enumerate(result.groups, start=1):
        names = ", ".join(result.teams[int(team)] for team in group)
        print(f"  {chr(64 + index)}: {names}")

    display = result.table.head(16).copy()
    for column in ("p_champion", "p_final", "p_sf", "p_qf", "p_r16"):
        display[column] = (100 * display[column]).round(1)
    display["se_champion"] = (100 * display["se_champion"]).round(2)

    print("\ntop 16 (percent):")
    print(display.to_string(index=False))

    csv_path = config.REPORTS_DIR / "simulation.csv"
    result.table.to_csv(csv_path, index=False)
    figure = simulate.plot_champions(
        result, config.FIGURES_DIR / "14_champion_probabilities.png"
    )
    print(f"\nwrote {csv_path.relative_to(config.PROJECT_ROOT)}")
    print(f"wrote {figure.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Football expected-goals model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="clean raw results and build datasets")
    p_build.add_argument("--synthetic", action="store_true",
                         help="generate a synthetic results.csv if none exists")
    p_build.add_argument("--force", action="store_true",
                         help="regenerate the synthetic CSV even if it exists")
    p_build.add_argument("--rename-former", action="store_true",
                         help="merge defunct nations into their successor")
    p_build.set_defaults(func=cmd_build)

    p_features = sub.add_parser("features", help="build the point-in-time feature table")
    p_features.set_defaults(func=cmd_features)

    p_baseline = sub.add_parser("baseline", help="fit baselines and score them")
    p_baseline.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    p_baseline.add_argument("--n-boot", type=int, default=1000,
                            help="bootstrap resamples for confidence intervals (0 = skip)")
    p_baseline.set_defaults(func=cmd_baseline)

    p_models = sub.add_parser("models", help="fit every model and compare")
    p_models.add_argument("--n-boot", type=int, default=1000,
                          help="bootstrap resamples for confidence intervals (0 = skip)")
    p_models.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    p_models.set_defaults(func=cmd_models)

    p_tune = sub.add_parser("tune", help="tune gradient boosting on validation")
    p_tune.set_defaults(func=cmd_tune)

    p_report = sub.add_parser("report", help="write reports/results.md and charts")
    p_report.add_argument("--n-boot", type=int, default=1000,
                          help="bootstrap resamples for confidence intervals")
    p_report.set_defaults(func=cmd_report)

    p_fit = sub.add_parser("fit", help="fit and pickle the models for prediction")
    p_fit.add_argument("--train-only", action="store_true",
                       help="fit on the train block only (matches the reported metrics)")
    p_fit.set_defaults(func=cmd_fit)

    p_predict = sub.add_parser("predict", help="predict a single fixture")
    p_predict.add_argument("--team-a", required=True)
    p_predict.add_argument("--team-b", required=True)
    p_predict.add_argument("--neutral", action=argparse.BooleanOptionalAction,
                           default=True, help="neutral venue (default: yes)")
    p_predict.add_argument("--tournament", default="FIFA World Cup")
    p_predict.add_argument("--model", default=predict.DEFAULT_MODEL,
                           choices=list(predict.MODEL_FACTORIES))
    p_predict.set_defaults(func=cmd_predict)

    p_simulate = sub.add_parser("simulate", help="Monte Carlo a 32-team World Cup")
    p_simulate.add_argument("--n-sims", type=int, default=simulate.N_SIMS)
    p_simulate.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_simulate.add_argument("--teams", default=None,
                            help="comma-separated 32 teams (default: top 32 by Elo)")
    p_simulate.add_argument("--model", default=predict.DEFAULT_MODEL,
                            choices=list(predict.MODEL_FACTORIES))
    p_simulate.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
