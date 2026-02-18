#!/usr/bin/env python3
"""Compatibility wrapper for unified PMC downloader."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from config_ingestion import IngestionConfig, ensure_data_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEGACY_REMOTE_DIR_TO_DATASET = {
    "/pub/pmc/oa_bulk/oa_comm/xml/": "pmc_oa",
    "/pub/pmc/manuscript/xml/": "author_manuscript",
}
DEFAULT_REMOTE_DIR = "/pub/pmc/oa_bulk/oa_comm/xml/"


def _load_unified_module():
    unified_path = Path(__file__).with_name("01_download_pmc_unified.py")
    spec = importlib.util.spec_from_file_location("download_pmc_unified", unified_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load unified downloader from {unified_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="[Deprecated] Download PMC bulk files")
    parser.add_argument("--output-dir", type=Path, default=IngestionConfig.PMC_XML_DIR)
    parser.add_argument("--remote-dir", type=str, default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    logger.warning(
        "scripts/01_download_pmc.py is deprecated. Use scripts/01_download_pmc_unified.py instead."
    )

    dataset = LEGACY_REMOTE_DIR_TO_DATASET.get(args.remote_dir)
    if dataset is None:
        raise ValueError(
            "Unsupported --remote-dir for compatibility wrapper. "
            f"Allowed values: {', '.join(LEGACY_REMOTE_DIR_TO_DATASET.keys())}"
        )

    ensure_data_dirs()
    unified = _load_unified_module()
    unified.download_pmc(output_dir=args.output_dir, datasets=[dataset], max_files=args.max_files)


if __name__ == "__main__":
    main()
