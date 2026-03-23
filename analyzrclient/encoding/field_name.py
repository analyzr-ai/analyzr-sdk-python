"""Sequential-key encoder for DataFrame column names.

Column names are replaced with short opaque keys (``X_0``, ``X_1``, …) before
data is sent to the analytics engine, preventing any information leakage through
column labels.  The bidirectional fref map enables full restoration on decode.
An expanded variant (fref_exp) handles one-hot dummy variable naming conventions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


class FieldNameEncoder:
    """Encodes/decodes DataFrame column names using sequential mapping (fref)."""

    @staticmethod
    def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode column names with sequential keys, generating a new fref key map.

        :param df: DataFrame whose column names will be obfuscated.
        :return: Tuple of (renamed DataFrame, fref dict with forward and reverse maps).
        :rtype: tuple[pd.DataFrame, dict]
        """
        df2 = deepcopy(df)
        fref: dict[str, dict[str, str]] = {
            "forward": {"PC_ID": "PC_ID"},
            "reverse": {"PC_ID": "PC_ID"},
        }
        cols2: list[str] = []
        for counter, col in enumerate(df2.columns):
            key = f"X_{counter}"
            fref["forward"][col] = key
            fref["reverse"][key] = col
            cols2.append(key)
        df2.columns = cols2
        return df2, fref

    @staticmethod
    def encode_with_keys(
        df: pd.DataFrame, fref: dict[str, dict[str, str]]
    ) -> pd.DataFrame:
        """Encode column names using an existing fref key map.

        :param df: DataFrame whose column names will be obfuscated.
        :param fref: Encoding keys dict with a ``forward`` map from original name to key.
        :return: Deep copy of the DataFrame with column names replaced by encoded keys.
        :rtype: pd.DataFrame
        """
        df2 = deepcopy(df)
        df2.columns = [fref["forward"][col] for col in df2.columns]
        return df2

    @staticmethod
    def to_expanded(
        fref: dict[str, dict[str, str]],
        xref: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Build an expanded fref covering one-hot dummy column names.

        For each categorical column in xref, creates entries for every
        ``{column}_{category}`` / ``{key}_{uuid}`` pair so that dummy columns
        produced by one-hot expansion can be decoded by name.

        :param fref: Base field-name encoding keys with forward and reverse maps.
        :param xref: Categorical encoding keys; used to enumerate dummy categories.
        :return: Expanded fref dict covering both plain and dummy column names.
        :rtype: dict
        """
        fref_exp: dict[str, dict[str, str]] = {"forward": {}, "reverse": {}}
        for col in fref["forward"]:
            key = fref["forward"][col]
            if col in xref:
                for category in xref[col]["forward"]:
                    col_exp = f"{col}_{category}"
                    key_exp = f"{key}_{xref[col]['forward'][category]}"
                    fref_exp["forward"][col_exp] = key_exp
                    fref_exp["reverse"][key_exp] = col_exp
            else:
                fref_exp["forward"][col] = key
                fref_exp["reverse"][key] = col
        return fref_exp

    @staticmethod
    def decode(df: pd.DataFrame, fref: dict[str, dict[str, str]]) -> pd.DataFrame:
        """Decode encoded column names back to original names.

        Returns the DataFrame unchanged when fref is empty.

        :param df: DataFrame with encoded column names.
        :param fref: Encoding keys dict with a ``reverse`` map from key to original name.
        :return: Deep copy of the DataFrame with original column names restored.
        :rtype: pd.DataFrame
        """
        if not fref:
            return df
        df2 = deepcopy(df)
        df2.columns = FieldNameEncoder.decode_columns(df2.columns, fref)
        return df2

    @staticmethod
    def decode_columns(cols: Any, fref: dict[str, dict[str, str]]) -> list[str]:
        """Decode an iterable of encoded column names to original names.

        :param cols: Iterable of encoded column name strings.
        :param fref: Encoding keys dict with a ``reverse`` map from key to original name.
        :return: List of decoded column name strings.
        :rtype: list[str]
        """
        if not fref:
            return list(cols)
        return [FieldNameEncoder.decode_value(col, fref) for col in cols]

    @staticmethod
    def decode_value(col: str, fref: dict[str, dict[str, str]]) -> str:
        """Decode encoded field name tags anywhere in the string.

        Tries exact match first, then replaces encoded keys found
        at any position. Longest keys are replaced first to prevent
        partial matches (e.g. X_22 before X_2).
        """
        if not fref:
            return col
        reverse = fref.get("reverse", {})
        if col in reverse:
            return reverse[col]
        result = col
        for key in sorted(reverse, key=len, reverse=True):
            if key in result:
                result = result.replace(key, reverse[key])
        return result
