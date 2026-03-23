"""Orchestration layer for encoding and decoding full DataFrames.

Composes all primitive encoders (categorical, numerical, boolean, field-name,
record-id) into a single stateless encode/decode interface used by analytics
runners before and after model execution.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import pandas as pd

from .boolean import BooleanEncoder
from .categorical import CategoricalEncoder
from .field_name import FieldNameEncoder
from .numerical import NumericalEncoder
from .record_id import RecordIdEncoder

log = logging.getLogger(__name__)


class DataEncoder:
    """Orchestrates the full encode/decode pipeline for input data.

    Composes CategoricalEncoder, NumericalEncoder, FieldNameEncoder,
    BooleanEncoder, and RecordIdEncoder into a single encode/decode interface.
    """

    @staticmethod
    def encode(
        df: pd.DataFrame,
        keys: dict[str, Any] | None = None,
        categorical_vars: list[str] | None = None,
        numerical_vars: list[str] | None = None,
        bool_vars: list[str] | None = None,
        skip_vars: list[str] | None = None,
        record_id_var: str | None = None,
        encode_field_names: bool = True,
        verbose: bool = False,
        numerical: bool = True,
    ) -> tuple[
        pd.DataFrame,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Encode a DataFrame, delegating to ``_encode_fresh`` or ``_encode_with_keys``.

        :param df: Input DataFrame to encode.
        :param keys: Existing encoding keys; when provided, encoding reuses prior mappings.
        :param categorical_vars: Column names to encode as categorical (UUID mapping).
        :param numerical_vars: Column names to encode as numerical (z-score normalization).
        :param bool_vars: Column names to encode as boolean (pass-through).
        :param skip_vars: Numerical columns to exclude from z-score normalization.
        :param record_id_var: Column name holding record identifiers to encode separately.
        :param encode_field_names: Whether to obfuscate column names with sequential keys.
        :param verbose: Emit progress log messages when ``True``.
        :param numerical: Apply z-score normalization when ``True``; pass through when ``False``.
        :return: Tuple of (encoded_df, xref, zref, rref, fref, fref_exp, bref).
        :rtype: tuple
        """
        if categorical_vars is None:
            categorical_vars = []
        if numerical_vars is None:
            numerical_vars = []
        if bool_vars is None:
            bool_vars = []
        if skip_vars is None:
            skip_vars = []

        if keys is None:
            return DataEncoder._encode_fresh(
                df,
                categorical_vars=categorical_vars,
                numerical_vars=numerical_vars,
                bool_vars=bool_vars,
                skip_vars=skip_vars,
                record_id_var=record_id_var,
                encode_field_names=encode_field_names,
                verbose=verbose,
                numerical=numerical,
            )
        return DataEncoder._encode_with_keys(
            df,
            keys,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
            bool_vars=bool_vars,
            skip_vars=skip_vars,
            record_id_var=record_id_var,
            encode_field_names=encode_field_names,
            verbose=verbose,
            numerical=numerical,
        )

    @staticmethod
    def decode(
        df: pd.DataFrame | None,
        categorical_vars: list[str] | None = None,
        numerical_vars: list[str] | None = None,
        bool_vars: list[str] | None = None,
        skip_vars: list[str] | None = None,
        record_id_var: str | None = None,
        xref: dict[str, Any] | None = None,
        zref: dict[str, Any] | None = None,
        rref: dict[str, Any] | None = None,
        fref: dict[str, Any] | None = None,
        bref: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame | None:
        """Decode a DataFrame using the encoding keys produced during a prior encode call.

        :param df: Encoded DataFrame to decode; returns as-is when ``None`` or empty.
        :param categorical_vars: Column names encoded as categorical to reverse-map.
        :param numerical_vars: Column names encoded as numerical to denormalize.
        :param bool_vars: Column names encoded as boolean to reverse-map.
        :param skip_vars: Numerical columns that were excluded from normalization.
        :param record_id_var: Column name of the encoded record identifier.
        :param xref: Categorical encoding keys (forward/reverse UUID maps per column).
        :param zref: Numerical encoding keys (mean/stdev per column).
        :param rref: Record ID encoding keys (forward/reverse UUID maps).
        :param fref: Field-name encoding keys (forward/reverse sequential maps).
        :param bref: Boolean encoding keys (currently unused pass-through).
        :param verbose: Emit progress log messages when ``True``.
        :return: Decoded DataFrame, or ``None`` if input was ``None``.
        :rtype: pd.DataFrame or None
        """
        if df is None or df.empty:
            return df
        if categorical_vars is None:
            categorical_vars = []
        if numerical_vars is None:
            numerical_vars = []
        if bool_vars is None:
            bool_vars = []
        if skip_vars is None:
            skip_vars = []
        if xref is None:
            xref = {}
        if zref is None:
            zref = {}
        if rref is None:
            rref = {}
        if fref is None:
            fref = {}
        if bref is None:
            bref = {}

        df2 = deepcopy(df)

        if verbose:
            log.info("Decoding field names...")
        df2 = FieldNameEncoder.decode(df2, fref)

        if verbose:
            log.info("Decoding categorical variables")
        for col in categorical_vars:
            if col in df2.columns:
                if verbose:
                    log.debug("  %s", col)
                df2[col] = CategoricalEncoder.decode(pd.Series(df2[col]), xref[col])

        if verbose:
            log.info("Decoding numerical variables")
        for col in numerical_vars:
            if col in df2.columns:
                if verbose:
                    log.debug("  %s", col)
                if col not in skip_vars:
                    df2[col] = NumericalEncoder.decode(
                        pd.Series(df2[col]).astype("float"), zref[col]
                    )

        for col in bool_vars:
            if col in df2.columns:
                if verbose:
                    log.debug("  %s", col)
                df2[col] = BooleanEncoder.decode(
                    pd.Series(df2[col]).astype("int"), bref[col]
                )

        if record_id_var is not None and record_id_var in df2.columns:
            if verbose:
                log.info("Decoding record IDs...")
            df2 = RecordIdEncoder.decode(df2, record_id_var, rref)

        return df2

    @staticmethod
    def _encode_fresh(
        df: pd.DataFrame,
        categorical_vars: list[str],
        numerical_vars: list[str],
        bool_vars: list[str],
        skip_vars: list[str],
        record_id_var: str | None,
        encode_field_names: bool,
        verbose: bool,
        numerical: bool,
    ) -> tuple[
        pd.DataFrame,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Encode a DataFrame from scratch, generating all encoding key maps.

        :param df: Source DataFrame.
        :param categorical_vars: Columns to UUID-encode.
        :param numerical_vars: Columns to z-score normalize.
        :param bool_vars: Columns to pass through as boolean.
        :param skip_vars: Numerical columns to exclude from normalization.
        :param record_id_var: Column name for record identifiers.
        :param encode_field_names: Obfuscate column names when ``True``.
        :param verbose: Emit progress log messages when ``True``.
        :param numerical: Apply z-score normalization when ``True``.
        :return: Tuple of (encoded_df, xref, zref, rref, fref, fref_exp, bref).
        :rtype: tuple
        """
        df2 = deepcopy(df)

        if verbose:
            log.info("Encoding categorical variables")
        xref: dict[str, Any] = {}
        for col in categorical_vars:
            if verbose:
                log.debug("  %s", col)
            df2[col], xref[col] = CategoricalEncoder.encode(pd.Series(df[col]))

        if verbose:
            log.info("Encoding numerical variables")
        zref: dict[str, Any] = {}
        for col in numerical_vars:
            if verbose:
                log.debug("  %s", col)
            if col not in skip_vars:
                df2[col], zref[col] = NumericalEncoder.encode(
                    pd.Series(df[col]), numerical=numerical
                )

        bref: dict[str, Any] = {}
        for col in bool_vars:
            if verbose:
                log.debug("  %s", col)
            df2[col], bref[col] = BooleanEncoder.encode(pd.Series(df[col]))

        rref: dict[str, Any] = {}
        if record_id_var is not None:
            if verbose:
                log.info("Encoding record IDs...")
            df2, rref = RecordIdEncoder.encode(df2, record_id_var)

        fref: dict[str, Any] = {}
        fref_exp: dict[str, Any] = {}
        if encode_field_names:
            if verbose:
                log.info("Encoding field names...")
            df2, fref = FieldNameEncoder.encode(df2)
            fref_exp = FieldNameEncoder.to_expanded(fref, xref)

        return df2, xref, zref, rref, fref, fref_exp, bref

    @staticmethod
    def _encode_with_keys(
        df: pd.DataFrame,
        keys: dict[str, Any],
        categorical_vars: list[str],
        numerical_vars: list[str],
        bool_vars: list[str],
        skip_vars: list[str],
        record_id_var: str | None,
        encode_field_names: bool,
        verbose: bool,
        numerical: bool,
    ) -> tuple[
        pd.DataFrame,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Encode a DataFrame using pre-existing encoding key maps.

        New categorical values not present in the original training set are added
        to the key maps and logged as warnings. Rows with any NaN are dropped after
        all encoding steps.

        :param df: Source DataFrame.
        :param keys: Encoding keys dict with xref, zref, bref, rref, fref, fref_exp entries.
        :param categorical_vars: Columns to UUID-encode using existing xref.
        :param numerical_vars: Columns to z-score normalize using existing zref.
        :param bool_vars: Columns to pass through using existing bref.
        :param skip_vars: Numerical columns to exclude from normalization.
        :param record_id_var: Column name for record identifiers.
        :param encode_field_names: Obfuscate column names when ``True``.
        :param verbose: Emit progress log messages when ``True``.
        :param numerical: Apply z-score normalization when ``True``.
        :return: Tuple of (encoded_df, xref, zref, rref, fref, fref_exp, bref).
        :rtype: tuple
        """
        df2 = deepcopy(df)

        if verbose:
            log.info("Encoding categorical variables")
        xref: dict[str, Any] = keys["xref"]
        for col in categorical_vars:
            if verbose:
                log.debug("  %s", col)
            df2[col], xref[col] = CategoricalEncoder.encode_with_keys(
                pd.Series(df[col]), xref[col]
            )

        if verbose:
            log.info("Encoding numerical variables")
        zref: dict[str, Any] = keys["zref"]
        for col in numerical_vars:
            if verbose:
                log.debug("  %s", col)
            if col not in skip_vars:
                df2[col] = NumericalEncoder.encode_with_keys(
                    pd.Series(df[col]), zref[col], numerical=numerical
                )

        bref: dict[str, Any] = keys["bref"]
        for col in bool_vars:
            if verbose:
                log.debug("  %s", col)
            df2[col] = BooleanEncoder.encode_with_keys(pd.Series(df[col]), bref[col])

        rref: dict[str, Any]
        if record_id_var is not None:
            if verbose:
                log.info("Encoding record IDs...")
            rref = keys["rref"]
            df2, rref = RecordIdEncoder.encode_with_keys(df2, record_id_var, rref)
        else:
            rref = {}

        fref: dict[str, Any]
        fref_exp: dict[str, Any]
        if encode_field_names:
            if verbose:
                log.info("Encoding field names...")
            fref = keys["fref"]
            fref_exp = keys["fref_exp"]
            df2 = FieldNameEncoder.encode_with_keys(df2, fref)
        else:
            fref = {}
            fref_exp = {}

        df2 = df2.dropna()
        return df2, xref, zref, rref, fref, fref_exp, bref
