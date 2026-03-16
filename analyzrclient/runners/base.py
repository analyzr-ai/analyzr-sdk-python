from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from ..auth.saml_sso import SamlSsoAuthClient
from ..encoding.data_encoder import DataEncoder
from ..encoding.domain import DomainCodec
from ..encoding.keys import KeyStore
from ..infrastructure.buffer import BufferClient
from ..infrastructure.polling import Poller


class BaseRunner:
    """Base class for all runners. Composes infrastructure, encoding, and codec services."""

    _client: SamlSsoAuthClient
    _base_url: str
    _buffer: BufferClient
    _poller: Poller
    _codec: DomainCodec | None

    def __init__(
        self, client: SamlSsoAuthClient | None = None,
        base_url: str | None = None,
        codec: DomainCodec | None = None,
    ) -> None:
        self._client = client  # type: ignore[assignment]
        self._base_url = base_url or ''
        self._buffer = BufferClient(self._client, self._base_url)
        self._poller = Poller(self._client, self._base_url)
        self._codec = codec

    @staticmethod
    def _get_request_id() -> str:
        return str(uuid.uuid4())

    @property
    def _analytics_uri(self) -> str:
        return f'{self._base_url}/analytics/'

    def _encode(
        self, df: pd.DataFrame, keys: dict[str, Any] | None = None,
        categorical_vars: list[str] | None = None,
        numerical_vars: list[str] | None = None,
        bool_vars: list[str] | None = None,
        skip_vars: list[str] | None = None,
        record_id_var: str | None = None,
        encode_field_names: bool = True,
        verbose: bool = False, numerical: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        return DataEncoder.encode(
            df, keys=keys, categorical_vars=categorical_vars,
            numerical_vars=numerical_vars, bool_vars=bool_vars,
            skip_vars=skip_vars, record_id_var=record_id_var,
            encode_field_names=encode_field_names, verbose=verbose,
            numerical=numerical,
        )

    def _decode(
        self, df: pd.DataFrame | None,
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
        return DataEncoder.decode(
            df, categorical_vars=categorical_vars,
            numerical_vars=numerical_vars, bool_vars=bool_vars,
            skip_vars=skip_vars, record_id_var=record_id_var,
            xref=xref, zref=zref, rref=rref, fref=fref, bref=bref,
            verbose=verbose,
        )

    def _read_frames(
        self, frame_names: list[str],
        request_id: str, client_id: str | None,
        verbose: bool = False, staging: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Read multiple named dataframes from the buffer."""
        return {
            name: self._buffer.read(
                request_id=request_id, client_id=client_id,
                dataframe_name=name, verbose=verbose, staging=staging,
            )
            for name in frame_names
        }

    def _keys_save(self, model_id: str, keys: dict[str, Any], verbose: bool = False) -> None:
        KeyStore.save(model_id, keys)

    def _keys_load(self, model_id: str, verbose: bool = False) -> dict[str, Any] | None:
        return KeyStore.load(model_id)
