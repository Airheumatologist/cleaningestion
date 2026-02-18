#!/usr/bin/env python3
"""Download PMC OA and Author Manuscript bulk files from NCBI FTP."""

from __future__ import annotations

import argparse
import ftplib
import gzip
import logging
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import requests

from config_ingestion import IngestionConfig, ensure_data_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
REMOTE_DIRS = {
    # PMC OA commercial subset. Full OA also includes oa_noncomm + oa_other.
    "pmc_oa": "/pub/pmc/oa_bulk/oa_comm/xml/",
    "author_manuscript": "/pub/pmc/manuscript/xml/",
}
ALLOWED_SUFFIXES = (".xml.gz", ".tar.gz")


def _iter_remote_files(ftp: ftplib.FTP, max_files: int | None = None) -> Iterable[str]:
    entries: list[str] = []
    ftp.retrlines("NLST", entries.append)

    matched = [f for f in sorted(entries) if any(f.endswith(s) for s in ALLOWED_SUFFIXES)]
    if max_files is not None:
        matched = matched[:max_files]
    yield from matched


def _download_file_http(remote_path: str, local_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download file via HTTPS from NCBI, returning True if successful."""
    url = f"https://ftp.ncbi.nlm.nih.gov{remote_path}"

    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        logger.info("Downloading via HTTP: %s", url)
        with requests.get(url, stream=True, timeout=60) as response:
            if response.status_code != 200:
                logger.warning("HTTP missing (status %s): %s", response.status_code, url)
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)

        temp_path.rename(local_path)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("HTTP download failed: %s", exc)
        return False
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_tar_gz(tar_path: Path, extract_dir: Path, delete_after: bool = True) -> int | None:
    """Extract .tar.gz archive and return extracted count, or None on failure."""
    extracted = 0
    logger.info("Extracting %s...", tar_path.name)

    tar_subdir = extract_dir / tar_path.stem.replace(".tar", "")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [
                member
                for member in tar.getmembers()
                if member.isfile() and (member.name.endswith(".nxml") or member.name.endswith(".xml"))
            ]

            tar_subdir.mkdir(parents=True, exist_ok=True)
            for member in members:
                tar.extract(member, path=tar_subdir)
                extracted_path = tar_subdir / member.name
                if extracted_path.exists():
                    final_path = extract_dir / Path(member.name).name
                    shutil.move(str(extracted_path), str(final_path))
                    extracted += 1

        if delete_after:
            tar_path.unlink(missing_ok=True)

        logger.info("Extracted %s files from %s", extracted, tar_path.name)
        return extracted
    except Exception as exc:
        logger.error("Failed to extract %s: %s", tar_path.name, exc)
        return None
    finally:
        shutil.rmtree(tar_subdir, ignore_errors=True)


def _extract_xml_gz(gz_path: Path, delete_after: bool = True) -> int | None:
    """Extract .xml.gz archive and return 1 on success, or None on failure."""
    output_path = gz_path.with_suffix("")
    try:
        logger.info("Extracting %s...", gz_path.name)
        with gzip.open(gz_path, "rb") as src:
            with open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        if delete_after:
            gz_path.unlink(missing_ok=True)

        logger.info("Extracted to %s", output_path.name)
        return 1
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        logger.error("Failed to extract %s: %s", gz_path.name, exc)
        return None


def _download_dataset(output_dir: Path, remote_dir: str, max_files: int | None = None) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ftp = None

    def _get_ftp() -> ftplib.FTP:
        nonlocal ftp
        if ftp is not None:
            try:
                ftp.voidcmd("NOOP")
                return ftp
            except Exception:
                ftp = None

        ftp = ftplib.FTP(FTP_HOST, timeout=120)
        ftp.login()
        ftp.cwd(remote_dir)
        return ftp

    try:
        conn = _get_ftp()
        files = list(_iter_remote_files(conn, max_files=max_files))
        logger.info("Found %s files in %s", len(files), remote_dir)
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:
                pass

    downloaded = 0
    extracted_count = 0

    def process_file(remote_name: str) -> tuple[int, int]:
        local_path = output_dir / remote_name
        marker_path = output_dir / f".{remote_name}.done"

        if marker_path.exists():
            logger.info("Skipping processed: %s", remote_name)
            return 0, 0

        downloaded_now = 0
        if not (local_path.exists() and local_path.stat().st_size > 0):
            remote_full_path = f"{remote_dir.rstrip('/')}/{remote_name}"
            if not _download_file_http(remote_full_path, local_path):
                logger.error("Failed to download %s", remote_name)
                return 0, 0
            downloaded_now = 1

        extracted_now = 0
        extraction_failed = False
        if remote_name.endswith(".tar.gz"):
            extracted = _extract_tar_gz(local_path, output_dir, delete_after=True)
            if extracted is None:
                extraction_failed = True
            else:
                extracted_now += extracted
        elif remote_name.endswith(".xml.gz"):
            extracted = _extract_xml_gz(local_path, delete_after=True)
            if extracted is None:
                extraction_failed = True
            else:
                extracted_now += extracted

        if extraction_failed:
            logger.warning("Not marking as done due to extraction failure: %s", remote_name)
            return downloaded_now, extracted_now

        marker_path.touch()
        return downloaded_now, extracted_now

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file, name): name for name in files}
        for index, future in enumerate(as_completed(future_to_file), start=1):
            remote_name = future_to_file[future]
            try:
                d_count, e_count = future.result()
                downloaded += d_count
                extracted_count += e_count
                logger.info(
                    "[%s/%s] Completed %s (downloaded=%s, extracted=%s)",
                    index,
                    len(files),
                    remote_name,
                    d_count,
                    e_count,
                )
            except Exception as exc:
                logger.error("%s generated an exception: %s", remote_name, exc)

    return downloaded, extracted_count


def download_pmc(
    output_dir: Path,
    datasets: List[str] | None = None,
    max_files: int | None = None,
) -> int:
    """Download requested datasets and return total files downloaded."""
    selected = datasets or list(REMOTE_DIRS.keys())
    invalid = [value for value in selected if value not in REMOTE_DIRS]
    if invalid:
        raise ValueError(f"Unknown dataset keys: {', '.join(invalid)}")

    total_downloaded = 0
    total_extracted = 0

    for dataset_key in selected:
        remote_dir = REMOTE_DIRS[dataset_key]
        dataset_output = output_dir / dataset_key
        dataset_output.mkdir(parents=True, exist_ok=True)
        (dataset_output / ".source").write_text(dataset_key, encoding="utf-8")

        logger.info("=" * 70)
        logger.info("Downloading %s from %s", dataset_key, remote_dir)
        logger.info("Destination: %s", dataset_output)

        downloaded, extracted = _download_dataset(dataset_output, remote_dir, max_files=max_files)
        total_downloaded += downloaded
        total_extracted += extracted

    logger.info("=" * 70)
    logger.info("Unified download complete. Downloaded files: %s", total_downloaded)
    logger.info("Unified extraction complete. Extracted XML/NXML files: %s", total_extracted)
    return total_downloaded


def _parse_datasets(value: str) -> List[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--datasets must include at least one dataset key")
    invalid = [item for item in datasets if item not in REMOTE_DIRS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown dataset(s): {', '.join(invalid)}. Allowed: {', '.join(REMOTE_DIRS.keys())}"
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PMC OA + Author Manuscript bulk files from NCBI FTP")
    parser.add_argument("--output-dir", type=Path, default=IngestionConfig.PMC_XML_DIR)
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=list(REMOTE_DIRS.keys()),
        help="Comma-separated dataset keys: pmc_oa,author_manuscript",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit files per dataset")
    args = parser.parse_args()

    ensure_data_dirs()
    download_pmc(output_dir=args.output_dir, datasets=args.datasets, max_files=args.max_files)


if __name__ == "__main__":
    main()
