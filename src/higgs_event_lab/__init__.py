"""Higgs Event Lab: Supervised Machine Learning Benchmark for HIGGS Dataset."""

__version__ = "0.1.0-dev"

from higgs_event_lab.artifacts import save_artifact_json
from higgs_event_lab.data import (
    HiggsBatch,
    extract_sample,
    iter_higgs_chunks,
    load_higgs_sample,
    stage_interim_clean_data,
    validate_raw_chunk,
)

from higgs_event_lab.evaluation import (
    compute_classification_metrics,
    select_operating_threshold,
)
from higgs_event_lab.features import (
    ALL_COLUMNS,
    ALL_FEATURES,
    FEATURE_GROUPS,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
    FeatureGroup,
    get_feature_columns,
)
from higgs_event_lab.classical import (
    LogisticRegressionConfig,
    build_pipeline as build_classical_pipeline,
    run_feature_group_benchmarks as run_classical_feature_group_benchmarks,
    train_logistic_regression,
    write_run_artifacts as write_classical_run_artifacts,
)
from higgs_event_lab.neural import (
    MLPConfig,
    build_mlp,
    run_feature_group_benchmarks,
    train_mlp,
    write_run_artifacts,
)
from higgs_event_lab.protocol import (
    ARTIFACT_SCHEMA_VERSION,
    SPLIT_PROTOCOL_VERSION,
    SplitConfig,
    SplitIndices,
    create_stratified_splits,
    new_experiment_artifact,
)

__all__ = [
    "ALL_COLUMNS",
    "ALL_FEATURES",
    "FEATURE_GROUPS",
    "HIGH_LEVEL_FEATURES",
    "LOW_LEVEL_FEATURES",
    "TARGET_COLUMN",
    "FeatureGroup",
    "get_feature_columns",
    "ARTIFACT_SCHEMA_VERSION",
    "SPLIT_PROTOCOL_VERSION",
    "SplitConfig",
    "SplitIndices",
    "create_stratified_splits",
    "new_experiment_artifact",
    "save_artifact_json",
    "compute_classification_metrics",
    "select_operating_threshold",
    "HiggsBatch",
    "extract_sample",
    "iter_higgs_chunks",
    "load_higgs_sample",
    "stage_interim_clean_data",
    "validate_raw_chunk",

    "LogisticRegressionConfig",
    "build_classical_pipeline",
    "train_logistic_regression",
    "write_classical_run_artifacts",
    "run_classical_feature_group_benchmarks",
    "MLPConfig",
    "build_mlp",
    "train_mlp",
    "write_run_artifacts",
    "run_feature_group_benchmarks",
]

