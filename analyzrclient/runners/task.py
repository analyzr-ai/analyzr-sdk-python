"""Runner for generic asynchronous test tasks used to validate API connectivity."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseRunner


class TaskRunner(BaseRunner):
    """Runs generic asynchronous test tasks against the Analyzr API.

    Primarily used for connectivity and smoke-testing; loads a fixed public
    dataset (Titanic CSV) and verifies the full request/buffer/poll round-trip.

    :param client: Authenticated SAML SSO client for API communication.
    :param base_url: Root URL of the Analyzr API tenant.
    """

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url)
        self._uri = f"{self._base_url}/analytics/"

    def run(
        self, task_type: str = "simple", verbose: bool = False, compressed: bool = False
    ) -> dict[str, Any]:
        """Submit a test task using the public Titanic dataset and return the processed result.

        :param task_type: Task variant to execute (e.g. ``'simple'``).
        :param verbose: Enable verbose polling output.
        :param compressed: Whether to compress the request payload.
        :return: Dict with ``status``, and a ``response`` containing the processed DataFrame
                 and metadata.
        :rtype: dict[str, Any]
        """
        request_id = self._get_request_id()
        df = pd.read_csv(
            "https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv",
            encoding="ISO-8859-1",
            low_memory=False,
        )
        return self._run(
            df=df,
            request_id=request_id,
            client_id="test",
            task_type=task_type,
            verbose=verbose,
            compressed=compressed,
        )

    def _run(
        self,
        df: pd.DataFrame,
        request_id: str,
        client_id: str = "test",
        task_type: str = "simple",
        verbose: bool = False,
        compressed: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "client_id": client_id,
            "type": task_type,
            "command": "test-task",
            "data": df.to_dict(),
        }
        self._client.post(self._uri, payload, compressed=compressed)
        self._poller.poll(
            payload={
                "request_id": request_id,
                "client_id": client_id,
                "command": "task-status",
            },
            timeout=60,
            step=1,
            verbose=verbose,
        )
        res = self._buffer.read(
            request_id=request_id, client_id=client_id, verbose=verbose, raw=True
        )
        if res["status"] != 200:
            return res
        return {
            "status": res["status"],
            "response": {
                "request_id": res["response"]["request_id"],
                "client_id": res["response"]["client_id"],
                "data": pd.DataFrame(res["response"]["data"]),
                "type": task_type,
            },
        }
