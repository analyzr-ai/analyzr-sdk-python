from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


class BooleanEncoder:
    """Encodes/decodes boolean values (pass-through — no transformation applied)."""

    @staticmethod
    def encode(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Pass-through encode, generating empty keys."""
        bref: dict[str, Any] = {}
        series2 = BooleanEncoder.encode_with_keys(series, bref)
        return series2, bref

    @staticmethod
    def encode_with_keys(series: pd.Series, bref: dict[str, Any]) -> pd.Series:
        """Pass-through encode with existing keys."""
        return deepcopy(series)

    @staticmethod
    def decode(series: pd.Series, bref: dict[str, Any]) -> pd.Series:
        """Pass-through decode."""
        return deepcopy(series)
