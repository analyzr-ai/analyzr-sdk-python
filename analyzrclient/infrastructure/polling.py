"""Long-poll client for monitoring asynchronous analytics task status.

Provides blocking and single-shot status inquiry operations against the
``/analytics/`` endpoint, waiting for a task to reach a terminal state
(``Complete`` or ``Failed``) or until a configurable timeout elapses.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from ..auth.saml_sso import SamlSsoAuthClient

log = logging.getLogger(__name__)


class Poller:
    """Polls task-status endpoint until completion or timeout.

    :param client: Authenticated API client used to issue status requests.
    :param base_url: Base URL of the analytics API (e.g. ``https://acme.api.g2m.ai/api/v1``).
    """

    _client: SamlSsoAuthClient
    _base_url: str

    def __init__(self, client: SamlSsoAuthClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url

    @property
    def _analytics_uri(self) -> str:
        return f"{self._base_url}/analytics/"

    def poll(
        self,
        payload: dict[str, Any] | None = None,
        timeout: int = 600,
        step: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Poll the task-status endpoint until the task reaches a terminal state or timeout.

        Swallows all exceptions per iteration; the last successful (or empty) response
        dict is returned when the loop exits.

        :param payload: Request payload to POST on each poll iteration.
        :param timeout: Maximum total seconds to wait before giving up.
        :param step: Seconds to sleep between each poll attempt.
        :param verbose: Write running status to stdout when ``True``.
        :return: Last API response dict (contains ``response.status`` on success).
        :rtype: dict
        """
        if payload is None:
            payload = {}
        counter = 0
        res: dict[str, Any] = {}
        while counter < timeout:
            time.sleep(step)
            try:
                res = self._client.post(self._analytics_uri, payload)
                if verbose:
                    sys.stdout.write(f"[poll][{counter}] {res}\r")
                status = res["response"]["status"]
                if status in ("Complete", "Failed") or "Failed:" in status:
                    break
            except Exception:
                pass
            counter += step
        if verbose:
            sys.stdout.write("\n")
        return res

    def status(
        self,
        request_id: str | None = None,
        client_id: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Perform a single task-status lookup for a specific request.

        Returns an empty dict when either identifier is missing, when the request
        raises an exception, or when the API returns a non-200 status code.

        :param request_id: Unique task request identifier.
        :param client_id: Client identifier scoping the request.
        :param verbose: Log error details when ``True``.
        :return: Task status response dict, or empty dict on failure.
        :rtype: dict
        """
        res: dict[str, Any] = {}
        if request_id is None or client_id is None:
            log.error(
                "Invalid status request (request_id=%s, client_id=%s)",
                request_id,
                client_id,
            )
        else:
            try:
                res = self._client.post(
                    self._analytics_uri,
                    {
                        "request_id": request_id,
                        "client_id": client_id,
                        "command": "task-status",
                    },
                )
            except Exception:
                if verbose:
                    log.error(
                        "Could not retrieve status for request_id=%s, client_id=%s",
                        request_id,
                        client_id,
                    )
            else:
                if res["status"] == 200:
                    res = res["response"]
                else:
                    if verbose:
                        log.error(
                            "Status inquiry for request_id=%s returned status=%s",
                            request_id,
                            res["status"],
                        )
                    res = {}
        return res
