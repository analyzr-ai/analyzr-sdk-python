from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseRunner


class TaskRunner(BaseRunner):
    """Runs pipeline for generic tasks (asynchronous testing)."""

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url)
        self._uri = f'{self._base_url}/analytics/'

    def run(self, type: str = 'simple', verbose: bool = False, compressed: bool = False) -> dict[str, Any]:
        """Run a test task against the API."""
        request_id = self._get_request_id()
        df = pd.read_csv(
            'https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv',
            encoding='ISO-8859-1', low_memory=False,
        )
        return self._run(df=df, request_id=request_id, client_id='test', type=type, verbose=verbose, compressed=compressed)

    def _run(
        self, df: pd.DataFrame, request_id: str,
        client_id: str = 'test', type: str = 'simple',
        verbose: bool = False, compressed: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'request_id': request_id,
            'client_id': client_id,
            'type': type,
            'command': 'test-task',
            'data': df.to_dict(),
        }
        self._client.post(self._uri, payload, compressed=compressed)
        self._poller.poll(
            payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
            timeout=60, step=1, verbose=verbose,
        )
        res = self._buffer.read(request_id=request_id, client_id=client_id, verbose=verbose, raw=True)
        if res['status'] != 200:
            return res
        return {
            'status': res['status'],
            'response': {
                'request_id': res['response']['request_id'],
                'client_id': res['response']['client_id'],
                'data': pd.DataFrame(res['response']['data']),
                'type': type,
            },
        }
