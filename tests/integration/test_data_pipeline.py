"""Integration tests for HIGGS sample extraction workflow (MID-01).

Uses synthetic fixtures to test end-to-end extraction, reproducibility, and manifest creation.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.data import (
    extract_sample,
    load_higgs_sample,
)


@pytest.fixture
def synthetic_higgs_gz(tmp_path: Path) -> Path:
    """Create a synthetic HIGGS compressed file with 200 rows."""
    source_file = tmp_path / "synthetic_HIGGS.csv.gz"
    rng = np.random.default_rng(999)
    # 200 rows, 29 columns
    labels = rng.integers(0, 2, size=(200, 1)).astype(np.float64)
    features = rng.normal(0, 1, size=(200, 28)).astype(np.float64)
    data = np.hstack([labels, features])

    with gzip.open(source_file, "wt", encoding="utf-8") as handle:
        for row in data:
            handle.write(",".join(f"{v:.6f}" for v in row) + "\n")
    return source_file


def test_extract_sample_sequential(synthetic_higgs_gz: Path, tmp_path: Path):
    out_file = tmp_path / "sample_50.csv.gz"
    manifest = extract_sample(
        source_path=synthetic_higgs_gz,
        output_path=out_file,
        n_rows=50,
        chunk_size=20,
        compare_later_chunk=True,
        later_chunk_offset_rows=100,
    )

    assert out_file.exists()
    assert manifest["rows_extracted"] == 50
    assert manifest["sampling_method"] == "sequential_head"
    assert manifest["random_seed"] is None
    assert manifest["label_counts"][0.0] + manifest["label_counts"][1.0] == 50

    # Verify extracted sample can be read by chunked loader
    sample = load_higgs_sample(out_file, n_rows=50, feature_group="all")
    assert sample.X.shape == (50, 28)
    assert sample.y.shape == (50,)


def test_extract_sample_seeded_reproducibility(synthetic_higgs_gz: Path, tmp_path: Path):
    out1 = tmp_path / "seeded_1.csv.gz"
    out2 = tmp_path / "seeded_2.csv.gz"

    extract_sample(
        source_path=synthetic_higgs_gz,
        output_path=out1,
        n_rows=40,
        chunk_size=20,
        seed=42,
        compare_later_chunk=False,
    )
    extract_sample(
        source_path=synthetic_higgs_gz,
        output_path=out2,
        n_rows=40,
        chunk_size=20,
        seed=42,
        compare_later_chunk=False,
    )

    sample1 = load_higgs_sample(out1, n_rows=40, feature_group="all")
    sample2 = load_higgs_sample(out2, n_rows=40, feature_group="all")

    assert np.array_equal(sample1.X.to_numpy(), sample2.X.to_numpy())
    assert np.array_equal(sample1.y.to_numpy(), sample2.y.to_numpy())


def test_extract_sample_cannot_overwrite_source(synthetic_higgs_gz: Path):
    with pytest.raises(ValueError, match="cannot overwrite source dataset"):
        extract_sample(
            source_path=synthetic_higgs_gz,
            output_path=synthetic_higgs_gz,
            n_rows=10,
        )


def test_extract_sample_creates_manifest(synthetic_higgs_gz: Path, tmp_path: Path):
    out_file = tmp_path / "sample_test.csv.gz"
    manifest = extract_sample(
        source_path=synthetic_higgs_gz,
        output_path=out_file,
        n_rows=25,
        chunk_size=10,
        seed=101,
        compare_later_chunk=False,
    )

    manifest_file = tmp_path / "sample_test.csv.gz.manifest.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["rows_extracted"] == 25
    assert data["random_seed"] == 101
    assert "source_sha256_prefix" in data
