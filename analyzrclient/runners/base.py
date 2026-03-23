"""Abstract base runner providing shared infrastructure for all analytics runners."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import pandas as pd

from ..auth.saml_sso import SamlSsoAuthClient
from ..encoding.data_encoder import DataEncoder
from ..encoding.domain import DomainCodec
from ..encoding.keys import KeyStore
from ..exceptions import AnalyzrError
from ..infrastructure.buffer import BufferClient
from ..infrastructure.polling import Poller

log = logging.getLogger(__name__)


class BaseRunner:
    """Base class for all analytics runners.

    Composes the authentication client, buffer, poller, encoding utilities,
    and an optional domain codec.  Concrete runner subclasses call
    ``super().__init__`` and may pass a codec tailored to their domain.

    :param client: Authenticated SAML SSO client used for all API calls.
    :param base_url: Root URL of the Analyzr API (e.g. ``https://tenant.example.com/api/v1``).
    :param codec: Optional domain-specific codec for encoding/decoding results.
    """

    _client: SamlSsoAuthClient
    _base_url: str
    _buffer: BufferClient
    _poller: Poller
    _codec: DomainCodec | None

    def __init__(
        self,
        client: SamlSsoAuthClient | None = None,
        base_url: str | None = None,
        codec: DomainCodec | None = None,
    ) -> None:
        self._client = client  # type: ignore[assignment]
        self._base_url = base_url or ""
        self._buffer = BufferClient(self._client, self._base_url)
        self._poller = Poller(self._client, self._base_url)
        self._codec = codec

    @staticmethod
    def _get_request_id() -> str:
        """Generate a unique request/model ID as a UUID4 string.

        :return: UUID4 string used as both a request identifier and model ID.
        :rtype: str
        """
        return str(uuid.uuid4())

    @property
    def _analytics_uri(self) -> str:
        """Return the analytics endpoint URI derived from the base URL.

        :return: Full analytics endpoint URL string.
        :rtype: str
        """
        return f"{self._base_url}/analytics/"

    def _encode(
        self,
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
        """Encode a DataFrame for transmission to the API.

        Delegates to :class:`DataEncoder` to obfuscate column names and encode
        categorical, numerical, and boolean variables.

        :param df: Raw input DataFrame.
        :param keys: Pre-existing encoding keys; when provided, encoding reuses
                     existing mappings (used during predict/run calls).
        :param categorical_vars: Column names to treat as categorical.
        :param numerical_vars: Column names to treat as numerical.
        :param bool_vars: Column names to treat as boolean.
        :param skip_vars: Column names to skip during value encoding.
        :param record_id_var: Column name used as the record identifier.
        :param encode_field_names: Whether to obfuscate column names.
        :param verbose: Enable verbose logging.
        :param numerical: Whether to apply numerical encoding.
        :return: Tuple of (encoded_df, xref, zref, rref, fref, fref_exp, bref)
                 where each ``*ref`` dict maps original values to encoded ones.
        :rtype: tuple
        """
        return DataEncoder.encode(
            df,
            keys=keys,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
            bool_vars=bool_vars,
            skip_vars=skip_vars,
            record_id_var=record_id_var,
            encode_field_names=encode_field_names,
            verbose=verbose,
            numerical=numerical,
        )

    def _decode(
        self,
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
        """Reverse-encode an API result DataFrame back to original column names and values.

        Delegates to :class:`DataEncoder` using the reference dicts produced by
        a prior :meth:`_encode` call.

        :param df: Encoded DataFrame returned from the API buffer; may be ``None``.
        :param categorical_vars: Categorical column names (original).
        :param numerical_vars: Numerical column names (original).
        :param bool_vars: Boolean column names (original).
        :param skip_vars: Column names that were skipped during encoding.
        :param record_id_var: Record identifier column name (original).
        :param xref: Categorical value cross-reference mapping.
        :param zref: Numerical z-score reference mapping.
        :param rref: Record ID reference mapping.
        :param fref: Field name forward/reverse reference mapping.
        :param bref: Boolean value reference mapping.
        :param verbose: Enable verbose logging.
        :return: Decoded DataFrame, or ``None`` if input was ``None``.
        :rtype: pd.DataFrame or None
        """
        return DataEncoder.decode(
            df,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
            bool_vars=bool_vars,
            skip_vars=skip_vars,
            record_id_var=record_id_var,
            xref=xref,
            zref=zref,
            rref=rref,
            fref=fref,
            bref=bref,
            verbose=verbose,
        )

    def _read_frames(
        self,
        frame_names: list[str],
        request_id: str,
        client_id: str | None,
        verbose: bool = False,
        staging: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Read multiple named DataFrames from the API result buffer.

        :param frame_names: List of logical frame names to retrieve.
        :param request_id: Unique identifier of the completed job.
        :param client_id: Client/tenant identifier.
        :param verbose: Enable verbose logging.
        :param staging: Whether to read from the staging buffer area.
        :return: Mapping of frame name to its corresponding DataFrame.
        :rtype: dict[str, pd.DataFrame]
        """
        return {
            name: self._buffer.read(
                request_id=request_id,
                client_id=client_id,
                dataframe_name=name,
                verbose=verbose,
                staging=staging,
            )
            for name in frame_names
        }

    def _keys_save(
        self, model_id: str, keys: dict[str, Any], verbose: bool = False
    ) -> None:
        """Persist encoding reference keys for a trained model to the local KeyStore.

        :param model_id: Model/request ID under which keys are stored.
        :param keys: Dict of reference mappings (xref, zref, rref, fref, fref_exp, bref).
        :param verbose: Enable verbose logging.
        """
        KeyStore.save(model_id, keys)

    def _keys_load(self, model_id: str, verbose: bool = False) -> dict[str, Any] | None:
        """Load persisted encoding reference keys for a trained model from the local KeyStore.

        :param model_id: Model/request ID whose keys should be retrieved.
        :param verbose: Enable verbose logging.
        :return: Dict of reference mappings, or ``None`` if no keys are found.
        :rtype: dict[str, Any] or None
        """
        return KeyStore.load(model_id)

    def delete_model(
        self,
        model_id: str,
        client_id: str,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Fully delete a trained model and all associated data.

        Removes the model blob, metadata record, and all associated staging data
        for the given model. The API handles model-type-specific cleanup (e.g.
        Iceberg tables for performance models) server-side via a strategy pattern.

        :param model_id: ID of the model to delete.
        :param client_id: Client/tenant identifier.
        :param verbose: Enable verbose logging.
        :return: API response dict with deletion summary.
        :rtype: dict[str, Any]
        :raises AnalyzrError: If the API returns a non-200 status.
        """
        if verbose:
            log.info("Deleting model and all associated data, model_id=%s", model_id)
        res = self._client.post(
            self._analytics_uri,
            {
                "command": "model-delete",
                "model_id": model_id,
                "client_id": client_id,
            },
        )
        if res["status"] not in (200, 201):
            log.error("Could not delete model: %s", res)
            raise AnalyzrError(
                "Could not delete model", detail=f"model_id={model_id}, response={res}"
            )
        if verbose:
            response = res.get("response", {})
            log.info(
                "Model deleted — cache entries cleared: %s",
                response.get("cache_entries_cleared", "unknown"),
            )
        return res["response"]
