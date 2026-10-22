"""Feature schema and column definitions for the HIGGS dataset.

Verified against the official UCI Machine Learning Repository (dataset id 280)
and docs/feature_dictionary.csv.

Layout:
    Column 0      : class_label (1.0 = signal, 0.0 = background)
    Columns 1-21  : 21 low-level kinematic features
    Columns 22-28 : 7 high-level derived invariant mass features
"""

from typing import Final, Literal

TARGET_COLUMN: Final[str] = "class_label"

LOW_LEVEL_FEATURES: Final[tuple[str, ...]] = (
    "lepton_pt",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet_1_pt",
    "jet_1_eta",
    "jet_1_phi",
    "jet_1_btag",
    "jet_2_pt",
    "jet_2_eta",
    "jet_2_phi",
    "jet_2_btag",
    "jet_3_pt",
    "jet_3_eta",
    "jet_3_phi",
    "jet_3_btag",
    "jet_4_pt",
    "jet_4_eta",
    "jet_4_phi",
    "jet_4_btag",
)

HIGH_LEVEL_FEATURES: Final[tuple[str, ...]] = (
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
)

ALL_FEATURES: Final[tuple[str, ...]] = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES

ALL_COLUMNS: Final[tuple[str, ...]] = (TARGET_COLUMN,) + ALL_FEATURES

FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "low_level": LOW_LEVEL_FEATURES,
    "high_level": HIGH_LEVEL_FEATURES,
    "all": ALL_FEATURES,
}

FeatureGroup = Literal["low_level", "high_level", "all"]


def get_feature_columns(group: FeatureGroup | str = "all") -> list[str]:
    """Return deterministic list of feature column names for the requested group.

    Guarantees that `class_label` (the target) is NEVER included.

    Parameters
    ----------
    group : {"low_level", "high_level", "all"}, default="all"
        The feature group to retrieve.

    Returns
    -------
    list[str]
        List of feature column names in deterministic UCI schema order.

    Raises
    ------
    ValueError
        If an unknown feature group is specified.
    """
    if group not in FEATURE_GROUPS:
        valid_groups = ", ".join(f"'{g}'" for g in sorted(FEATURE_GROUPS))
        raise ValueError(f"Unknown feature group '{group}'. Expected one of: {valid_groups}")
    cols = list(FEATURE_GROUPS[group])
    assert TARGET_COLUMN not in cols, "Safety invariant violated: target in feature list"
    return cols
