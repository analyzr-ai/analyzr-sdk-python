"""HTTP client for interacting with the Analyzr data buffer service.

Handles batched DataFrame uploads with per-batch retry, typed reads (DataFrame,
JSON, raw), and housekeeping operations (clear, purge, usage, queue purge).
All writes use the ``/buffer/`` endpoint; queue purge targets ``/analytics/``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from io import StringIO
from typing import Any

import pandas as pd

from ..auth.saml_sso import SamlSsoAuthClient

log = logging.getLogger(__name__)


class BufferClient:
    """Manages data buffer operations: save, read, clear, purge, usage.

    :param client: Authenticated API client used to issue all buffer requests.
    :param base_url: Base URL of the analytics API (e.g. ``https://acme.api.g2m.ai/api/v1``).
    """

    _client: SamlSsoAuthClient
    _base_url: str

    def __init__(self, client: SamlSsoAuthClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url

    @property
    def _buffer_uri(self) -> str:
        return f"{self._base_url}/buffer/"

    @property
    def _analytics_uri(self) -> str:
        return f"{self._base_url}/analytics/"

    def save(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000,
        client_id: str | None = None,
        request_id: str | None = None,
        verbose: bool = False,
        compressed: bool = False,
        staging: bool = False,
    ) -> dict[str, Any]:
        """Save a DataFrame to the buffer in row batches with per-batch retry.

        Each batch is attempted up to five times with a 2-second back-off.
        Returns immediately with null batch counts when ``request_id`` is ``None``.

        :param df: DataFrame to upload.
        :param batch_size: Number of rows per upload batch.
        :param client_id: Client identifier scoping the buffer slot.
        :param request_id: Unique request identifier for this upload session.
        :param verbose: Write batch progress to stdout when ``True``.
        :param compressed: Send compressed payload when ``True``.
        :param staging: Use CSV upload format instead of dict when ``True``.
        :return: Dict with ``request_id``, ``total_batches``, and ``batches_saved`` counts.
        :rtype: dict
        """
        if verbose:
            log.info("Saving data to buffer...")
        if request_id is None:
            log.warning("Request ID is None, aborting buffer save.")
            return {"request_id": None, "total_batches": None, "batches_saved": None}

        batched_df: list[pd.DataFrame] = [
            df.iloc[i : i + batch_size] for i in range(0, len(df), batch_size)
        ]
        success = 0
        for idx, batch in enumerate(batched_df, 1):
            for attempt in range(5):
                if self._batch_save(
                    batch,
                    idx,
                    len(batched_df),
                    client_id,
                    request_id,
                    verbose=(attempt == 4),
                    compressed=compressed,
                    staging=staging,
                ):
                    if verbose:
                        sys.stdout.write(
                            f"Processed batch {idx} of {len(batched_df)}\r"
                        )
                    success += 1
                    break
                else:
                    time.sleep(2)
        if verbose:
            sys.stdout.write("\n")

        return {
            "request_id": request_id,
            "total_batches": len(batched_df),
            "batches_saved": success,
        }

    def _batch_save(
        self,
        batch: pd.DataFrame,
        idx: int,
        n: int,
        client_id: str | None,
        request_id: str,
        verbose: bool = False,
        compressed: bool = False,
        staging: bool = False,
    ) -> bool:
        """Upload a single batch to the buffer."""
        try:
            res = self._client.post(
                self._buffer_uri,
                {
                    "command": "upload",
                    "data": batch.to_dict()
                    if not staging
                    else batch.to_csv(header=(idx == 1), index=False),
                    "request_id": request_id,
                    "client_id": client_id,
                    "staging": staging,
                },
                compressed=compressed,
            )
            if res["status"] == 200:
                return True
            if verbose:
                log.warning(
                    "Returned status %s for batch %s of %s", res["status"], idx, n
                )
        except Exception:
            if verbose:
                log.warning("API call failed for batch %s of %s", idx, n)
        return False

    def read(
        self,
        client_id: str | None = None,
        request_id: str | None = None,
        dataframe_name: str = "df",
        verbose: bool = False,
        raw: bool = False,
        staging: bool = False,
        dataframe: bool = True,
    ) -> Any:
        """Read data from the buffer, returning it in the requested format.

        :param client_id: Client identifier scoping the buffer slot.
        :param request_id: Unique request identifier for the target buffer slot.
        :param dataframe_name: Named DataFrame key within the buffer slot.
        :param verbose: Log error details when ``True``.
        :param raw: Return the raw API response dict when ``True``; overrides all other format flags.
        :param staging: Parse response as CSV when ``True``; as dict otherwise.
        :param dataframe: Return a ``pd.DataFrame`` when ``True``; JSON-decoded dict when ``False``.
        :return: DataFrame, dict, or raw response depending on flag combination.
        """
        res2 = self._read(
            client_id=client_id,
            request_id=request_id,
            dataframe_name=dataframe_name,
            verbose=verbose,
            staging=staging,
        )
        if raw:
            return res2
        if res2["status"] != 200:
            log.error("Buffer read failed: %s", res2)
            return {} if not dataframe else pd.DataFrame(None)
        if res2["response"]["data"] is None or res2["response"]["data"] == "":
            return {} if not dataframe else pd.DataFrame()
        if not dataframe:
            return json.loads(res2["response"]["data"])
        if staging:
            try:
                return pd.read_csv(StringIO(res2["response"]["data"]))
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        return pd.DataFrame(res2["response"]["data"])

    def _read(
        self,
        client_id: str | None = None,
        request_id: str | None = None,
        dataframe_name: str = "df",
        verbose: bool = False,
        staging: bool = False,
    ) -> dict[str, Any]:
        return self._client.post(
            self._buffer_uri,
            {
                "command": "read",
                "request_id": request_id,
                "client_id": client_id,
                "dataframe_name": dataframe_name,
                "staging": staging,
            },
        )

    def clear(
        self,
        client_id: str | None = None,
        request_id: str | None = None,
        verbose: bool = False,
        out_of_core: bool = False,
    ) -> dict[str, Any]:
        """Clear the buffer slot for a specific request.

        :param client_id: Client identifier scoping the buffer slot.
        :param request_id: Unique request identifier for the buffer slot to clear.
        :param verbose: Log the clear operation when ``True``.
        :param out_of_core: Signal to the server that out-of-core mode was used when ``True``.
        :return: API response dict.
        :rtype: dict
        """
        if verbose:
            log.info("Clearing buffer (out_of_core=%s)", out_of_core)
        res = self._client.post(
            self._buffer_uri,
            {
                "command": "clear",
                "request_id": request_id,
                "client_id": client_id,
                "out_of_core": out_of_core,
            },
        )
        if res["status"] != 200:
            log.warning("Buffer clear failed: %s", res)
        return res

    def purge(
        self, client_id: str | None = None, verbose: bool = False
    ) -> dict[str, Any]:
        """Purge all buffer data for a client across all request slots.

        :param client_id: Client identifier whose entire buffer should be purged.
        :param verbose: Log the purge operation when ``True``.
        :return: API response dict.
        :rtype: dict
        """
        if verbose:
            log.info("Purging buffer...")
        res = self._client.post(
            self._buffer_uri,
            {
                "command": "purge",
                "client_id": client_id,
            },
        )
        if res["status"] != 200:
            log.warning("Buffer purge failed: %s", res)
        return res

    def usage(
        self, client_id: str | None = None, verbose: bool = False
    ) -> dict[str, Any]:
        """Check buffer storage usage for a client.

        :param client_id: Client identifier to query.
        :param verbose: Log the usage query when ``True``.
        :return: API response dict containing usage statistics.
        :rtype: dict
        """
        if verbose:
            log.info("Checking buffer usage...")
        res = self._client.post(
            self._buffer_uri,
            {
                "command": "usage",
                "client_id": client_id,
            },
        )
        if res["status"] != 200:
            log.warning("Checking buffer usage failed: %s", res)
        return res

    def queue_purge(
        self, client_id: str | None = None, verbose: bool = False
    ) -> dict[str, Any]:
        """Purge the analytics task queue for a client.

        Targets the ``/analytics/`` endpoint rather than ``/buffer/``.

        :param client_id: Client identifier whose queue should be purged.
        :param verbose: Log the purge operation when ``True``.
        :return: API response dict.
        :rtype: dict
        """
        if verbose:
            log.info("Purging queue...")
        res = self._client.post(
            self._analytics_uri,
            {
                "command": "purge",
                "client_id": client_id,
            },
        )
        if res["status"] != 200:
            log.warning("Queue purge failed: %s", res)
        return res
