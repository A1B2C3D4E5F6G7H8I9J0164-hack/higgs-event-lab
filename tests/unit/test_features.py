"""Unit tests for feature definitions and schema helpers."""

import csv
import sys
from pathlib import Path

import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.features import (
    ALL_COLUMNS,
    ALL_FEATURES,
    FEATURE_GROUPS,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
    get_feature_columns,
)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
DICTIONARY_PATH = DOCS_DIR / "feature_dictionary.csv"


def test_target_column_name():
    assert TARGET_COLUMN == "class_label"


def test_feature_counts():
    assert len(LOW_LEVEL_FEATURES) == 21
    assert len(HIGH_LEVEL_FEATURES) == 7
    assert len(ALL_FEATURES) == 28
    assert len(ALL_COLUMNS) == 29
    assert ALL_COLUMNS[0] == TARGET_COLUMN
    assert ALL_COLUMNS[1:] == ALL_FEATURES


def test_target_column_never_in_feature_groups():
    for group_name, cols in FEATURE_GROUPS.items():
        assert TARGET_COLUMN not in cols, f"{TARGET_COLUMN} must never be in feature group {group_name}"


def test_get_feature_columns_returns_correct_subsets():
    low = get_feature_columns("low_level")
    assert len(low) == 21
    assert low == list(LOW_LEVEL_FEATURES)
    assert TARGET_COLUMN not in low

    high = get_feature_columns("high_level")
    assert len(high) == 7
    assert high == list(HIGH_LEVEL_FEATURES)
    assert TARGET_COLUMN not in high

    all_cols = get_feature_columns("all")
    assert len(all_cols) == 28
    assert all_cols == list(ALL_FEATURES)
    assert TARGET_COLUMN not in all_cols


def test_get_feature_columns_invalid_group_raises():
    with pytest.raises(ValueError, match="Unknown feature group"):
        get_feature_columns("invalid_group_name")


def test_features_match_feature_dictionary_csv():
    with open(DICTIONARY_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["name"] == TARGET_COLUMN
    assert rows[0]["group"] == "target"

    dict_low = [r["name"] for r in rows[1:22]]
    dict_high = [r["name"] for r in rows[22:29]]

    assert list(LOW_LEVEL_FEATURES) == dict_low
    assert list(HIGH_LEVEL_FEATURES) == dict_high
    assert list(ALL_FEATURES) == dict_low + dict_high
