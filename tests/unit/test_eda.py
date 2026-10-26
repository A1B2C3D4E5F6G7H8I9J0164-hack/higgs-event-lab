import numpy as np
import pandas as pd

from higgs_event_lab.eda import save_eda_outputs
from higgs_event_lab.features import ALL_FEATURES


def test_eda_outputs_save_report_tables_and_figures(tmp_path):
    rows = 20
    X = pd.DataFrame(np.arange(rows * len(ALL_FEATURES), dtype=float).reshape(rows, len(ALL_FEATURES)), columns=ALL_FEATURES)
    y = np.array([0, 1] * 10)

    report = save_eda_outputs(X, y, output_dir=tmp_path, data_source="synthetic")

    assert report["rows"] == rows
    assert report["feature_count"] == 28
    assert report["label_counts"] == {"0": 10, "1": 10}
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "feature_summary.csv").is_file()
    assert (tmp_path / "correlation_heatmap.png").is_file()
