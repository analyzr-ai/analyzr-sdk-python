from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .categorical import CategoricalEncoder


class RecordIdEncoder:
    """Encodes/decodes record ID columns (delegates to CategoricalEncoder)."""

    @staticmethod
    def encode(
        df: pd.DataFrame, record_id_var: str,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode record ID column, generating new keys."""
        df2 = deepcopy(df)
        df2[record_id_var], rref = CategoricalEncoder.encode(pd.Series(df[record_id_var]))
        return df2, rref

    @staticmethod
    def encode_with_keys(
        df: pd.DataFrame, record_id_var: str, rref: dict[str, dict[str, str]],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode record ID column using existing keys."""
        df2 = deepcopy(df)
        df2[record_id_var], rref = CategoricalEncoder.encode_with_keys(pd.Series(df[record_id_var]), rref)
        return df2, rref

    @staticmethod
    def decode(
        df: pd.DataFrame, record_id_var: str, rref: dict[str, dict[str, str]],
    ) -> pd.DataFrame:
        """Decode record ID column back to original values."""
        if not rref:
            return df
        df2 = deepcopy(df)
        if record_id_var not in df2.columns:
            df2[record_id_var] = df2.index
        series: pd.Series = pd.Series(df[record_id_var])
        df2[record_id_var] = CategoricalEncoder.decode(series, rref)
        df2 = df2.set_index(record_id_var)
        return df2
