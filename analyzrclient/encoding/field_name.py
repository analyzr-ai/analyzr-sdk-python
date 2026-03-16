from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


class FieldNameEncoder:
    """Encodes/decodes DataFrame column names using sequential mapping (fref)."""

    @staticmethod
    def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode column names, generating new keys."""
        df2 = deepcopy(df)
        fref: dict[str, dict[str, str]] = {
            'forward': {'PC_ID': 'PC_ID'},
            'reverse': {'PC_ID': 'PC_ID'},
        }
        cols2: list[str] = []
        for counter, col in enumerate(df2.columns):
            key = f'X_{counter}'
            fref['forward'][col] = key
            fref['reverse'][key] = col
            cols2.append(key)
        df2.columns = cols2
        return df2, fref

    @staticmethod
    def encode_with_keys(df: pd.DataFrame, fref: dict[str, dict[str, str]]) -> pd.DataFrame:
        """Encode column names using existing keys."""
        df2 = deepcopy(df)
        df2.columns = [fref['forward'][col] for col in df2.columns]
        return df2

    @staticmethod
    def to_expanded(
        fref: dict[str, dict[str, str]], xref: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Convert fref to expanded field names using dummy variable convention."""
        fref_exp: dict[str, dict[str, str]] = {'forward': {}, 'reverse': {}}
        for col in fref['forward']:
            key = fref['forward'][col]
            if col in xref:
                for category in xref[col]['forward']:
                    col_exp = f'{col}_{category}'
                    key_exp = f'{key}_{xref[col]["forward"][category]}'
                    fref_exp['forward'][col_exp] = key_exp
                    fref_exp['reverse'][key_exp] = col_exp
            else:
                fref_exp['forward'][col] = key
                fref_exp['reverse'][key] = col
        return fref_exp

    @staticmethod
    def decode(df: pd.DataFrame, fref: dict[str, dict[str, str]]) -> pd.DataFrame:
        """Decode column names back to original."""
        if not fref:
            return df
        df2 = deepcopy(df)
        df2.columns = FieldNameEncoder.decode_columns(df2.columns, fref)
        return df2

    @staticmethod
    def decode_columns(cols: Any, fref: dict[str, dict[str, str]]) -> list[str]:
        """Decode array of column names."""
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
        reverse = fref.get('reverse', {})
        if col in reverse:
            return reverse[col]
        result = col
        for key in sorted(reverse, key=len, reverse=True):
            if key in result:
                result = result.replace(key, reverse[key])
        return result
