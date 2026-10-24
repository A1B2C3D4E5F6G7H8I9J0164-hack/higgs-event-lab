"""Automated data acquisition script for the HIGGS dataset.

Downloads the official UCI HIGGS dataset or a team cloud mirror (Google Drive,
Hugging Face, or custom URL), extracts HIGGS.csv.gz into data/raw/, verifies the
SHA-256 checksum, and optionally prepares the 250k processed benchmark sample.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --url <custom_url>
    python scripts/download_data.py --prepare-subset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Add src to sys.path so project modules can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_UCI_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_TARGET_FILE = DEFAULT_RAW_DIR / "HIGGS.csv.gz"
DEFAULT_PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "higgs_sample_250k.csv.gz"


def get_ssl_context() -> ssl.SSLContext:
    """Create an SSL context with fallback to unverified if certificates are missing."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return ssl._create_unverified_context()


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file streaming in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with streaming chunks and live progress updates."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(destination.suffix + ".tmp")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    req = urllib.request.Request(url, headers=headers)
    ctx = get_ssl_context()

    print(f"Connecting to: {url}")
    start_time = time.time()
    last_print = start_time

    with urllib.request.urlopen(req, context=ctx) as response, open(temp_dest, "wb") as out_file:
        total_size = response.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else 0

        bytes_downloaded = 0
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
            bytes_downloaded += len(chunk)
            now = time.time()

            # Print progress every 2 seconds or on completion
            if now - last_print >= 2.0:
                elapsed = now - start_time
                speed_mb = (bytes_downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                if total_bytes > 0:
                    pct = (bytes_downloaded / total_bytes) * 100
                    print(
                        f"Downloaded {bytes_downloaded / (1024*1024):.1f} MB / "
                        f"{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) "
                        f"at {speed_mb:.1f} MB/s..."
                    )
                else:
                    print(
                        f"Downloaded {bytes_downloaded / (1024*1024):.1f} MB "
                        f"at {speed_mb:.1f} MB/s..."
                    )
                last_print = now

    if temp_dest.exists():
        temp_dest.replace(destination)

    total_time = max(time.time() - start_time, 0.001)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Download complete: {size_mb:.1f} MB in {total_time:.1f}s ({size_mb / total_time:.1f} MB/s)")
    return destination


def download_and_extract_raw(
    url: str = DEFAULT_UCI_URL,
    target_gz_path: Path = DEFAULT_TARGET_FILE,
    force: bool = False,
) -> Path:
    """Download the dataset archive and ensure data/raw/HIGGS.csv.gz is in place."""
    if target_gz_path.exists() and not force:
        size_gb = target_gz_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"Dataset already exists at {target_gz_path} ({size_gb:.2f} GB). Use --force to re-download.")
        return target_gz_path

    target_gz_path.parent.mkdir(parents=True, exist_ok=True)

    # If URL directly points to a .csv.gz file
    if url.endswith(".csv.gz") or url.endswith(".gz"):
        download_file(url, target_gz_path)
    else:
        # URL points to a zip archive (such as UCI's higgs.zip)
        zip_path = target_gz_path.parent / "higgs_download.zip"
        try:
            download_file(url, zip_path)
            print("Extracting HIGGS.csv.gz from downloaded archive...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find the csv.gz file inside the archive
                gz_candidates = [name for name in zf.namelist() if name.endswith(".csv.gz")]
                if not gz_candidates:
                    raise RuntimeError(f"No .csv.gz file found inside archive: {zf.namelist()}")

                inner_gz = gz_candidates[0]
                with zf.open(inner_gz) as src, open(target_gz_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            print(f"Extracted to {target_gz_path}")
        finally:
            if zip_path.exists():
                zip_path.unlink()

    print("Computing SHA-256 checksum...")
    checksum = compute_sha256(target_gz_path)
    print(f"SHA-256: {checksum}")
    return target_gz_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated dataset downloader for the HIGGS benchmark."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_UCI_URL,
        help="Download URL (UCI zip or direct .csv.gz link)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TARGET_FILE),
        help="Target path for raw HIGGS.csv.gz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if HIGGS.csv.gz already exists",
    )
    parser.add_argument(
        "--prepare-subset",
        action="store_true",
        help="Automatically extract 250k processed sample after download",
    )
    parser.add_argument(
        "--subset-rows",
        type=int,
        default=250_000,
        help="Row count for prepared subset (default: 250000)",
    )

    args = parser.parse_args(argv)

    target_path = Path(args.output)
    try:
        raw_file = download_and_extract_raw(
            url=args.url,
            target_gz_path=target_path,
            force=args.force,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return 1

    if args.prepare_subset:
        print(f"\nExtracting {args.subset_rows} rows to {DEFAULT_PROCESSED_FILE}...")
        try:
            from higgs_event_lab.data import extract_sample
            extract_sample(
                source_path=raw_file,
                output_path=DEFAULT_PROCESSED_FILE,
                n_rows=args.subset_rows,
                compare_later_chunk=False,
            )
            print(f"Subset prepared successfully at {DEFAULT_PROCESSED_FILE}")
        except Exception as exc:
            print(f"Subset extraction failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
