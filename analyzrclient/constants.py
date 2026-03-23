"""Package-wide constants and configuration defaults for the Analyzr SDK."""

from __future__ import annotations

from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

VERBOSE: bool = True

try:
    _resolved_version = version("analyzr")
except PackageNotFoundError:
    _resolved_version = "dev"
CLIENT_VERSION: str = _resolved_version
HOME: str = str(Path.home())
TEMP_DIR: str = "{}/.analyzr".format(HOME)

REGRESSION_DEFAULT_ALGO: str = "linear-regression"
