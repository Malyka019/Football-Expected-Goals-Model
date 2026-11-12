"""Split tests: boundaries, partitioning, and the failure modes of a bad split."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import splits  # noqa: E402


def make(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "match_id": range(len(dates)),
        "tournament_category": "friendly",
    })


def test_split_boundaries_are_inclusive_and_chronological():
    df = make([
        "2010-01-01", "2017-12-31",   # train (last day inclusive)
        "2018-01-01", "2021-12-31",   # validation
        "2022-01-01", "2022-12-31",   # test
        "2023-01-01",                 # unused (real data runs past 2022)
    ])
    labelled = splits.assign_split(df)

    assert labelled["split"].tolist() == [
        "train", "train", "validation", "validation",
        "test", "test", "unused",
    ]
    splits.validate_splits(labelled)


def test_split_frames_partitions_every_row_exactly_once():
    df = splits.assign_split(make([f"{y}-06-01" for y in range(2010, 2024)]))
    frames = splits.split_frames(df)

    assert set(frames) == {"train", "validation", "test", "unused"}
    total = sum(len(block) for block in frames.values())
    assert total == len(df)

    ids = [set(block["match_id"]) for block in frames.values()]
    assert sum(len(s) for s in ids) == len(set().union(*ids)), "a match is in two splits"


# Every block needs at least one row, otherwise the "empty split" check fires
# first and the test would pass for the wrong reason.
FULL_BLOCKS = ["2010-01-01", "2015-01-01", "2019-06-01", "2022-06-01"]


def test_validate_catches_a_training_row_labelled_as_test():
    df = splits.assign_split(make(FULL_BLOCKS))
    df.loc[df.index[0], "split"] = "test"   # time travel
    with pytest.raises(ValueError, match="overlaps"):
        splits.validate_splits(df)


def test_validate_catches_a_match_in_two_splits():
    df = splits.assign_split(make(FULL_BLOCKS))
    # Same match_id appears as both train and test.
    extra = df.iloc[[0]].copy()
    extra["split"] = "test"
    df = pd.concat([df, extra], ignore_index=True)

    with pytest.raises(ValueError, match="more than one split"):
        splits.validate_splits(df)


def test_validate_rejects_unlabelled_rows():
    df = splits.assign_split(make(["2010-01-01"]))
    df["split"] = None
    with pytest.raises(ValueError):
        splits.validate_splits(df)


def test_world_cup_slice_follows_the_tournament_window():
    df = make([
        "2022-06-01",   # qualifier-ish, outside the window
        "2022-11-19",   # day before
        "2022-11-20",   # opening match
        "2022-12-18",   # final
        "2022-12-19",   # day after
    ])
    df["tournament_category"] = "world_cup"

    # A friendly played in the same weeks must not inflate the tournament count.
    stray = make(["2022-12-02"])
    stray["tournament_category"] = "friendly"
    stray["match_id"] = 99
    df = pd.concat([df, stray], ignore_index=True)

    wc = splits.world_cup_slice(df, 2022)
    assert len(wc) == 2, "only the two World Cup fixtures count"
    assert (wc["tournament_category"] == "world_cup").all()
    assert wc["date"].min() == pd.Timestamp("2022-11-20")
    assert wc["date"].max() == pd.Timestamp("2022-12-18")


def test_describe_splits_reports_ranges():
    df = splits.assign_split(make(["2010-01-01", "2019-01-01", "2022-06-01"]))
    described = splits.describe_splits(df)

    assert set(described["split"]) == {"train", "validation", "test"}
    assert described.set_index("split").loc["train", "start"] == pd.Timestamp("2010-01-01").date()
    assert described["rows"].sum() == 3
