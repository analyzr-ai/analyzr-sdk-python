"""Pass-through encoder for boolean columns.

Boolean values require no transformation before or after model execution.
The encoder preserves the interface contract expected by DataEncoder while
generating an empty key map (bref) for consistency with other encoder types.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


class BooleanEncoder:
    """Encodes/decodes boolean values (pass-through — no transformation applied)."""

    @staticmethod
    def encode(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Pass-through encode, generating an empty bref key map.

        :param series: Boolean series to encode.
        :return: Tuple of (deep-copied series, empty bref dict).
        :rtype: tuple[pd.Series, dict]
        """
        bref: dict[str, Any] = {}
        series2 = BooleanEncoder.encode_with_keys(series, bref)
        return series2, bref

    @staticmethod
    def encode_with_keys(series: pd.Series, bref: dict[str, Any]) -> pd.Series:
        """Pass-through encode with existing keys; bref is ignored.

        :param series: Boolean series to encode.
        :param bref: Existing boolean encoding keys (unused).
        :return: Deep copy of the input series.
        :rtype: pd.Series
        """
        return deepcopy(series)

    @staticmethod
    def decode(series: pd.Series, bref: dict[str, Any]) -> pd.Series:
        """Pass-through decode; bref is ignored.

        :param series: Encoded boolean series to decode.
        :param bref: Boolean encoding keys (unused).
        :return: Deep copy of the input series.
        :rtype: pd.Series
        """
        return deepcopy(series)
