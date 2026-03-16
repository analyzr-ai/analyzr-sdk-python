from __future__ import annotations

import logging
import sys
import time
from typing import Any

from ..auth.saml_sso import SamlSsoAuthClient

log = logging.getLogger(__name__)


class Poller:
    """Polls task-status endpoint until completion or timeout."""

    _client: SamlSsoAuthClient
    _base_url: str

    def __init__(self, client: SamlSsoAuthClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url

    @property
    def _analytics_uri(self) -> str:
        return f'{self._base_url}/analytics/'

    def poll(
        self, payload: dict[str, Any] | None = None,
        timeout: int = 600, step: int = 1, verbose: bool = False,
    ) -> dict[str, Any]:
        """Poll task-status until Complete, Failed, or timeout."""
        if payload is None:
            payload = {}
        counter = 0
        res: dict[str, Any] = {}
        while counter < timeout:
            time.sleep(step)
            try:
                res = self._client.post(self._analytics_uri, payload)
                if verbose:
                    sys.stdout.write(f'[poll][{counter}] {res}\r')
                status = res['response']['status']
                if status in ('Complete', 'Failed') or 'Failed:' in status:
                    break
            except Exception:
                pass
            counter += step
        if verbose:
            sys.stdout.write('\n')
        return res

    def status(
        self, request_id: str | None = None,
        client_id: str | None = None, verbose: bool = False,
    ) -> dict[str, Any]:
        """Check status for a specific request."""
        res: dict[str, Any] = {}
        if request_id is None or client_id is None:
            log.error('Invalid status request (request_id=%s, client_id=%s)', request_id, client_id)
        else:
            try:
                res = self._client.post(self._analytics_uri, {
                    'request_id': request_id,
                    'client_id': client_id,
                    'command': 'task-status',
                })
            except Exception:
                if verbose:
                    log.error('Could not retrieve status for request_id=%s, client_id=%s', request_id, client_id)
            else:
                if res['status'] == 200:
                    res = res['response']
                else:
                    if verbose:
                        log.error('Status inquiry for request_id=%s returned status=%s', request_id, res['status'])
                    res = {}
        return res
