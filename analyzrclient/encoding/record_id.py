"""Record-identifier encoder wrapping CategoricalEncoder for row-ID columns.

Record IDs are treated as opaque categorical values and encoded with UUIDs via
CategoricalEncoder.  On decode the original ID values are restored and the column
is promoted to the DataFrame index.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .categorical import CategoricalEncoder


class RecordIdEncoder:
    """Encodes/decodes record ID columns (delegates to CategoricalEncoder)."""

    @staticmethod
    def encode(
        df: pd.DataFrame,
        record_id_var: str,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode the record ID column with UUIDs, generating a new rref key map.

        :param df: DataFrame containing the record ID column.
        :param record_id_var: Name of the column holding record identifiers.
        :return: Tuple of (DataFrame with UUID-encoded IDs, rref dict).
        :rtype: tuple[pd.DataFrame, dict]
        """
        df2 = deepcopy(df)
        df2[record_id_var], rref = CategoricalEncoder.encode(
            pd.Series(df[record_id_var])
        )
        return df2, rref

    @staticmethod
    def encode_with_keys(
        df: pd.DataFrame,
        record_id_var: str,
        rref: dict[str, dict[str, str]],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encode the record ID column with UUIDs using an existing rref key map.

        :param df: DataFrame containing the record ID column.
        :param record_id_var: Name of the column holding record identifiers.
        :param rref: Existing encoding keys (mutated in-place for unseen IDs).
        :return: Tuple of (DataFrame with UUID-encoded IDs, updated rref dict).
        :rtype: tuple[pd.DataFrame, dict]
        """
        df2 = deepcopy(df)
        df2[record_id_var], rref = CategoricalEncoder.encode_with_keys(
            pd.Series(df[record_id_var]), rref
        )
        return df2, rref

    @staticmethod
    def decode(
        df: pd.DataFrame,
        record_id_var: str,
        rref: dict[str, dict[str, str]],
    ) -> pd.DataFrame:
        """Decode the record ID column back to original values and set it as the index.

        When rref is empty the DataFrame is returned unchanged.  When the record ID
        column is absent it is populated from the existing integer index before decoding.

        :param df: DataFrame with UUID-encoded record IDs.
        :param record_id_var: Name of the record ID column to decode.
        :param rref: Encoding keys dict with a ``reverse`` map from UUID to original ID.
        :return: DataFrame with original record IDs restored as the index.
        :rtype: pd.DataFrame
        """
        if not rref:
            return df
        df2 = deepcopy(df)
        if record_id_var not in df2.columns:
            df2[record_id_var] = df2.index
        series: pd.Series = pd.Series(df[record_id_var])
        df2[record_id_var] = CategoricalEncoder.decode(series, rref)
        df2 = df2.set_index(record_id_var)
        return df2
