import csv
from pathlib import Path

DICTIONARY = Path(__file__).resolve().parents[2] / "docs" / "feature_dictionary.csv"

UCI_ORDER = [
    "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi",
    "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag",
    "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag",
    "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag",
    "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]


def load_rows():
    with open(DICTIONARY, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalise(uci_name):
    return uci_name.lower().replace("-", "").replace(" ", "_")


def test_dictionary_has_label_plus_28_features():
    rows = load_rows()
    assert len(rows) == 29
    assert rows[0]["group"] == "target"
    assert rows[0]["name"] == "class_label"


def test_group_sizes_are_21_and_7():
    groups = [r["group"] for r in load_rows()[1:]]
    assert groups.count("low_level") == 21
    assert groups.count("high_level") == 7


def test_low_level_columns_come_before_high_level_columns():
    groups = [r["group"] for r in load_rows()[1:]]
    assert groups == ["low_level"] * 21 + ["high_level"] * 7


def test_file_column_indices_are_contiguous_from_zero():
    indices = [int(r["file_column_index"]) for r in load_rows()]
    assert indices == list(range(29))


def test_model_input_indices_skip_the_label():
    rows = load_rows()
    assert rows[0]["model_input_index"] == ""
    assert [int(r["model_input_index"]) for r in rows[1:]] == list(range(28))


def test_names_are_unique_and_snake_case():
    names = [r["name"] for r in load_rows()]
    assert len(names) == len(set(names))
    for name in names:
        assert name == name.lower() and " " not in name


def test_feature_order_matches_uci_documentation():
    names = [r["name"] for r in load_rows()[1:]]
    assert names == [normalise(n) for n in UCI_ORDER]


def test_every_row_has_a_description():
    for row in load_rows():
        assert row["description"].strip()
