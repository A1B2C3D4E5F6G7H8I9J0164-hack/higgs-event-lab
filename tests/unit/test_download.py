"""Unit tests for the automated data acquisition script (scripts/download_data.py)."""

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_data import (
    compute_sha256,
    download_and_extract_raw,
    get_ssl_context,
    main,
)


def test_get_ssl_context():
    """Verify that an SSL context can be instantiated cleanly."""
    ctx = get_ssl_context()
    assert ctx is not None


def test_compute_sha256(tmp_path: Path):
    """Verify streaming SHA-256 computation against hashlib reference."""
    test_file = tmp_path / "sample.bin"
    content = b"Simulated particle collision event stream 12345"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    actual_hash = compute_sha256(test_file)
    assert actual_hash == expected_hash


def test_existing_raw_file_skips_download(tmp_path: Path):
    """Verify that download_and_extract_raw skips downloading if file exists and force=False."""
    target_file = tmp_path / "HIGGS.csv.gz"
    target_file.write_bytes(b"mock gz content")

    result = download_and_extract_raw(
        url="https://dummy.example.com/higgs.zip",
        target_gz_path=target_file,
        force=False,
    )
    assert result == target_file
    assert target_file.read_bytes() == b"mock gz content"


def test_cli_help(capsys):
    """Verify CLI help option."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
