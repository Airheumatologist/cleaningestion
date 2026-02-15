#!/usr/bin/env python3
"""Download PMC Open Access bulk files from NCBI FTP (no AWS dependency)."""

from __future__ import annotations

import argparse
import ftplib
import gzip
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Iterable

from config_ingestion import IngestionConfig, ensure_data_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import requests

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
DEFAULT_REMOTE_DIR = "/pub/pmc/oa_bulk/oa_comm/xml/"
S3_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
ALLOWED_SUFFIXES = (".xml.gz", ".tar.gz")


def _iter_remote_files(ftp: ftplib.FTP, max_files: int | None = None) -> Iterable[str]:
    """List remote files matching ALLOWED_SUFFIXES, optionally limited to max_files."""
    entries: list[str] = []
    ftp.retrlines("NLST", entries.append)

    matched = [f for f in sorted(entries) if any(f.endswith(s) for s in ALLOWED_SUFFIXES)]
    if max_files is not None:
        matched = matched[:max_files]
    yield from matched


def _download_file_http(remote_path: str, local_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download file via HTTP (S3), returning True if successful."""
    # Map FTP path to S3 URL: /pub/pmc/oa_bulk/oa_comm/xml/ -> /oa_comm/xml/
    # Effectively remove /pub/pmc/oa_bulk from start if present
    rel_path = remote_path.replace("/pub/pmc/oa_bulk", "")
    url = f"{S3_BASE_URL}{rel_path}"
    
    try:
        logger.info("Downloading via HTTP: %s", url)
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 200:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                return True
            else:
                logger.warning("HTTP missing (status %s): %s", r.status_code, url)
    except Exception as e:
        logger.warning("HTTP download failed: %s", e)
    
    return False


def _download_file_ftp(ftp: ftplib.FTP, remote_name: str, local_path: Path, chunk_size: int = 1024 * 1024) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info("Skipping existing file: %s", local_path.name)
        return

    logger.info("Downloading via FTP %s", remote_name)
    with open(local_path, "wb") as handle:
        ftp.retrbinary(f"RETR {remote_name}", handle.write, blocksize=chunk_size)


def _extract_tar_gz(tar_path: Path, extract_dir: Path, delete_after: bool = True) -> int:
    """Extract a .tar.gz file to the specified directory. Returns number of files extracted."""
    extracted = 0
    logger.info("Extracting %s...", tar_path.name)
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Extract all .nxml and .xml files
            members = [m for m in tar.getmembers() if m.isfile() and (m.name.endswith('.nxml') or m.name.endswith('.xml'))]
            for member in members:
                # Extract to a subdirectory named after the tar file (without .tar.gz)
                tar_subdir = extract_dir / tar_path.stem.replace('.tar', '')
                tar_subdir.mkdir(parents=True, exist_ok=True)
                
                # Extract the file
                tar.extract(member, path=tar_subdir)
                
                # Move to main directory if nested
                extracted_path = tar_subdir / member.name
                if extracted_path.exists():
                    final_path = extract_dir / Path(member.name).name
                    shutil.move(str(extracted_path), str(final_path))
                    extracted += 1
            
            # Clean up empty subdir
            tar_subdir = extract_dir / tar_path.stem.replace('.tar', '')
            if tar_subdir.exists():
                shutil.rmtree(tar_subdir, ignore_errors=True)
        
        if delete_after:
            logger.info("Removing archive %s after extraction", tar_path.name)
            tar_path.unlink(missing_ok=True)
        
        logger.info("Extracted %s files from %s", extracted, tar_path.name)
        return extracted
    except Exception as e:
        logger.error("Failed to extract %s: %s", tar_path.name, e)
        return 0


def _extract_xml_gz(gz_path: Path, delete_after: bool = True) -> Path | None:
    """Extract a .xml.gz file to .xml. Returns path to extracted file."""
    output_path = gz_path.with_suffix('')  # Remove .gz
    
    try:
        logger.info("Extracting %s...", gz_path.name)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        if delete_after:
            logger.info("Removing archive %s after extraction", gz_path.name)
            gz_path.unlink(missing_ok=True)
        
        logger.info("Extracted to %s", output_path.name)
        return output_path
    except Exception as e:
        logger.error("Failed to extract %s: %s", gz_path.name, e)
        return None


def download_pmc(output_dir: Path, remote_dir: str, max_files: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to %s for listing", FTP_HOST)
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(remote_dir)

    downloaded = 0
    extracted_count = 0
    try:
        files = list(_iter_remote_files(ftp, max_files=max_files))
        logger.info("Found %s files in %s", len(files), remote_dir)
        
        for idx, remote_name in enumerate(files, start=1):
            logger.info("[%s/%s] %s", idx, len(files), remote_name)
            local_path = output_dir / remote_name
            
            # Check existence first
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.info("Skipping existing: %s", remote_name)
            else:
                # Try HTTP first
                remote_full_path = f"{remote_dir.rstrip('/')}/{remote_name}"
                if _download_file_http(remote_full_path, local_path):
                    downloaded += 1
                else:
                    # Fallback to FTP
                    logger.info("Falling back to FTP for %s", remote_name)
                    _download_file_ftp(ftp, remote_name, local_path)
                    downloaded += 1
            
            # Extract downloaded archives
            if local_path.exists() and local_path.stat().st_size > 0:
                if remote_name.endswith('.tar.gz'):
                    extracted = _extract_tar_gz(local_path, output_dir, delete_after=True)
                    extracted_count += extracted
                elif remote_name.endswith('.xml.gz'):
                    extracted_file = _extract_xml_gz(local_path, delete_after=True)
                    if extracted_file:
                        extracted_count += 1
    finally:
        ftp.quit()

    logger.info("Download complete. Files downloaded: %s", downloaded)
    logger.info("Extraction complete. XML files extracted: %s", extracted_count)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PMC OA bulk files from NCBI FTP")
    parser.add_argument("--output-dir", type=Path, default=IngestionConfig.PMC_XML_DIR)
    parser.add_argument("--remote-dir", type=str, default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    ensure_data_dirs()
    download_pmc(output_dir=args.output_dir, remote_dir=args.remote_dir, max_files=args.max_files)


if __name__ == "__main__":
    main()
