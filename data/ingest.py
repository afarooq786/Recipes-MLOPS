#!/usr/bin/env python3
"""
data/ingest.py — Step 1: Data Ingestion (dataset download script)

Downloads the recipe dataset from Kaggle
(thedevastator/better-recipes-for-a-better-life) and writes the raw CSV
files into data/raw/, so the rest of the pipeline (validate.py, split.py)
has a stable local copy to work from.

Authentication:
    Reads a Kaggle API token from the KAGGLE_API_TOKEN environment
    variable (kagglehub also falls back to ~/.kaggle/access_token or the
    legacy ~/.kaggle/kaggle.json automatically — see kagglehub docs).
    Generate a token at https://www.kaggle.com/settings/api -> API Tokens
    -> Generate, then:

        export KAGGLE_API_TOKEN=<your token>

Usage:
    python data/ingest.py
    python data/ingest.py --force                 # re-download even if raw files already exist
    python data/ingest.py --dataset <owner/slug> --dest data/raw
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

DEFAULT_DATASET = "thedevastator/better-recipes-for-a-better-life"
DEFAULT_DEST = Path(__file__).resolve().parent / "raw"

# Files we expect this specific dataset to contain. Used to sanity-check
# the download and to short-circuit re-downloading when they're already
# present locally.
EXPECTED_FILES = ["recipes.csv", "test_recipes.csv"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def _have_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if Path("~/.kaggle/access_token").expanduser().exists():
        return True
    if Path("~/.kaggle/access_token.txt").expanduser().exists():
        return True
    if Path("~/.kaggle/kaggle.json").expanduser().exists():
        return True
    return False


def _check_credentials() -> None:
    if not _have_credentials():
        logger.error(
            "No Kaggle credentials found. Set the KAGGLE_API_TOKEN environment variable "
            "(generate a token at https://www.kaggle.com/settings/api -> API Tokens -> Generate) "
            "before running this script."
        )
        sys.exit(1)


def download_dataset(dataset: str, dest: Path, force: bool = False) -> Path:
    """Download `dataset` from Kaggle into `dest`. Idempotent unless force=True."""
    dest.mkdir(parents=True, exist_ok=True)

    already_present = all((dest / f).exists() for f in EXPECTED_FILES)
    if already_present and not force:
        logger.info("Raw files already present in %s — skipping download (use --force to re-download).", dest)
        return dest

    _check_credentials()

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    logger.info("Downloading dataset '%s' from Kaggle...", dataset)
    try:
        cache_path = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:  # noqa: BLE001 - surface any auth/network error clearly
        logger.error("Failed to download dataset '%s' from Kaggle: %s", dataset, exc)
        raise
    logger.info("Downloaded to local kagglehub cache: %s", cache_path)

    copied = []
    for csv_file in sorted(cache_path.glob("*.csv")):
        target = dest / csv_file.name
        shutil.copy2(csv_file, target)
        copied.append(target.name)

    if not copied:
        logger.error("No CSV files found in the downloaded dataset at %s", cache_path)
        sys.exit(1)

    logger.info("Copied %d file(s) into %s: %s", len(copied), dest, ", ".join(copied))

    missing = [f for f in EXPECTED_FILES if not (dest / f).exists()]
    if missing:
        logger.warning(
            "Expected file(s) not found after download: %s. The Kaggle dataset contents may have "
            "changed — double check DEFAULT_DATASET / EXPECTED_FILES in this script.",
            missing,
        )

    return dest


def summarize(dest: Path) -> None:
    import pandas as pd

    for f in EXPECTED_FILES:
        path = dest / f
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            logger.info("%s: %d rows, %d columns", f, len(df), len(df.columns))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not summarize %s: %s", f, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the recipe dataset from Kaggle.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug to download.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Directory to write raw CSV files to.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    args = parser.parse_args()

    dest = Path(args.dest)
    download_dataset(args.dataset, dest, force=args.force)
    summarize(dest)
    logger.info("Ingestion complete. Raw files are in %s", dest)


if __name__ == "__main__":
    main()
