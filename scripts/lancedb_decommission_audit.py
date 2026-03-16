#!/usr/bin/env python3
"""Phase 8 decommission audit for residual Qdrant dependencies."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_EXCLUDES = [
    "scripts/05_setup_qdrant.py",
    "src/retriever_qdrant.py",
    "ingestionplan.md",
    "README.md",
]


def _collect_matches(repo_root: Path, excludes: list[str]) -> list[dict[str, str]]:
    if shutil.which("rg"):
        cmd = ["rg", "-n", "qdrant|Qdrant", "src", "scripts", "tests", "-S"]
    else:
        cmd = ["grep", "-R", "-n", "-E", "qdrant|Qdrant", "src", "scripts", "tests"]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]

    matches: list[dict[str, str]] = []
    for line in lines:
        path = line.split(":", 1)[0]
        if any(path.startswith(exclude) for exclude in excludes):
            continue
        matches.append({"location": line})
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit codebase for remaining Qdrant runtime coupling")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matches = _collect_matches(repo_root, DEFAULT_EXCLUDES)
    status = "ok" if not matches else "blocked"

    result = {
        "status": status,
        "blocked": bool(matches),
        "blockers": matches,
        "emergency_restore_snapshot_required": True,
        "next_steps": [
            "Remove retriever_qdrant imports from runtime modules",
            "Delete Qdrant-only setup scripts after burn-in",
            "Retain one emergency snapshot and restore drill notes",
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()
