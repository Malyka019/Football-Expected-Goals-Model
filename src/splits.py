"""Time-based train / validation / test splits (step 3).

Why not `train_test_split(shuffle=True)`
----------------------------------------
Shuffling a football history is not a modelling shortcut, it is a different
(and much easier) problem: a random test match in 2012 gets to be predicted by a
model that has seen 2019. It also leaks through the features, because Elo and
form are built from past matches only -- so the model learns "trust the rating"
in a world where the rating always had its future available. The result looks
great and predicts nothing.

So the split is a strict prefix/suffix partition of the calendar:

    train       ... <= 2017-12-31     (fit coefficients)
    validation  2018-01-01..2021-12-31 (tune, compare, choose)
    test        2022-01-01..2022-12-31 (untouched until step 6)
    unused      > 2022-12-31           (real data runs to the present)

The 2022 World Cup sits inside the test block; the 2018 one inside validation.
Headline "World Cup only" numbers are reported separately with
`world_cup_slice`, because the tournament is the thing we actually care about
and burying 64 matches inside a year of friendlies hides the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SPLIT_LABELS = ("train", "validation", "test", "unused")


def assign_split(
    df: pd.DataFrame,
    *,
    train_end: str = config.TRAIN_END,
    val_start: str = config.VAL_START,
    val_end: str = config.VAL_END,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> pd.DataFrame:
    """Add a `split` column: train / validation / test / unused.

    Boundaries are inclusive on both ends. Works on either the match-level or
    the team-level table, since both carry a `date` column.
    """
    dates = pd.to_datetime(df["date"])
    train_end_ts = pd.Timestamp(train_end)
    val_start_ts, val_end_ts = pd.Timestamp(val_start), pd.Timestamp(val_end)
    test_start_ts, test_end_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)

    split = np.select(
        [
            dates <= train_end_ts,
            (dates >= val_start_ts) & (dates <= val_end_ts),
            (dates >= test_start_ts) & (dates <= test_end_ts),
        ],
        ["train", "validation", "test"],
        default="unused",
    )
    out = df.copy()
    out["split"] = pd.Categorical(split, categories=SPLIT_LABELS, ordered=True)
    return out


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {label: subset} for the non-empty splits, preserving row order."""
    if "split" not in df.columns:
        df = assign_split(df)
    return {
        label: df[df["split"] == label].copy()
        for label in SPLIT_LABELS
        if (df["split"] == label).any()
    }


def validate_splits(df: pd.DataFrame) -> None:
    """Raise if the split is not a strict chronological partition.

    Cheap to run, and it catches the two ways a split silently goes wrong:
    an off-by-one that puts a match in two blocks, and a manual override that
    leaves a training match sitting after a test match.
    """
    if "split" not in df.columns:
        raise ValueError("no 'split' column: call assign_split first")

    if df["split"].isna().any():
        raise ValueError("some rows have no split label")

    bounds = {}
    for label in ("train", "validation", "test"):
        block = df.loc[df["split"] == label, "date"]
        if block.empty:
            raise ValueError(f"split '{label}' is empty")
        bounds[label] = (block.min(), block.max())

    # Checked before the ordering test: it gives a more precise diagnosis when
    # the same match has been assigned twice.
    if "match_id" in df.columns:
        per_match = df.groupby("match_id")["split"].nunique()
        if (per_match > 1).any():
            raise ValueError("a match appears in more than one split")

    # The blocks must not overlap in time.
    if not bounds["train"][1] < bounds["validation"][0]:
        raise ValueError("training data overlaps or follows validation data")
    if not bounds["validation"][1] < bounds["test"][0]:
        raise ValueError("validation data overlaps or follows test data")


def describe_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts, date ranges and label composition per split."""
    if "split" not in df.columns:
        df = assign_split(df)
    rows = []
    for label in SPLIT_LABELS:
        block = df[df["split"] == label]
        if block.empty:
            continue
        rows.append({
            "split": label,
            "rows": len(block),
            "matches": block["match_id"].nunique() if "match_id" in block else np.nan,
            "start": block["date"].min().date(),
            "end": block["date"].max().date(),
            "world_cup_rows": int(
                (block.get("tournament_category") == "world_cup").sum()
            ) if "tournament_category" in block else 0,
        })
    return pd.DataFrame(rows)


def world_cup_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rows from the World Cup itself, inside its tournament window.

    The date filter alone is not enough: the real data contains a handful of
    unrelated internationals played during the same weeks (and a 2022 window that
    would otherwise return 87 matches for a 64-match tournament). Filtering on
    the tournament category as well keeps the headline number honest.
    """
    start, end = config.WORLD_CUP_WINDOWS[year]
    dates = pd.to_datetime(df["date"])
    window = df[(dates >= start) & (dates <= end)]
    if "tournament_category" in window.columns:
        window = window[window["tournament_category"] == "world_cup"]
    return window.copy()
