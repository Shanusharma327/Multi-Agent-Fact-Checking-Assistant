"""Put the source tree on the import path for the test suite.

The code lives under ``src/`` so the repository follows the layout the
course asks for, but the packages are imported by their own names
(``fact_checker``, and the analysis modules by bare module name). pytest
adds the rootdir, not ``src/``, so the two directories are added here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "src", ROOT / "src" / "analysis"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
