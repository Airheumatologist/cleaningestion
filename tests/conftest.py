from __future__ import annotations

import importlib.util


collect_ignore_glob: list[str] = []

if importlib.util.find_spec("lancedb") is None:
    collect_ignore_glob.extend(
        [
            "test_lancedb_*.py",
        ]
    )

