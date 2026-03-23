"""HTTP basic-auth client for the Analyzr API.

Provides GET and POST operations authenticated with username/password credentials.
Intended for environments where SAML SSO is unavailable (e.g. service accounts,
automated testing).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

log = logging.getLogger(__name__)


class BasicAuthClient:
    """Manages low-level interactions with the Analyzr API using basic authentication.

    :param user: API username.
    :param pwd: API password.
    :param host: API hostname without protocol (e.g. ``acme.api.g2m.ai``).
        Terminates the process with an error message when ``None``.
    """

    def __init__(self, user: str = "", pwd: str = "", host: str | None = None) -> None:
        if host is None:
            log.error("Please provide a valid host, e.g. host=acme.api.g2m.ai")
            exit(1)
        self._user: str = user
        self._pwd: str = pwd
        self._base_url: str = f"https://{host}/api/v1"

    def get(self, uri: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send an authenticated GET request.

        :param uri: Fully qualified URL to request.
        :param params: Optional query parameters to append to the request.
        :return: Normalized response dict with ``status`` and ``response`` keys.
        :rtype: dict
        """
        if params is None:
            r = requests.get(uri, auth=(self._user, self._pwd))
        else:
            r = requests.get(uri, auth=(self._user, self._pwd), params=params)
        return self._response(r)

    def post(self, uri: str, json_obj: Any, compressed: bool = False) -> dict[str, Any]:
        """Send an authenticated POST request with a JSON-serialized body.

        :param uri: Fully qualified URL to POST to.
        :param json_obj: Payload to serialize and send as the request body.
        :param compressed: Reserved for compressed upload support (currently unused).
        :return: Normalized response dict with ``status`` and ``response`` keys.
        :rtype: dict
        """
        data = json.dumps(json_obj)
        r = requests.post(uri, auth=(self._user, self._pwd), json=data)
        return self._response(r)

    def _response(self, r: requests.Response) -> dict[str, Any]:
        res: dict[str, Any] = {"status": r.status_code}
        if 200 <= r.status_code < 300:
            res["response"] = r.json()
        else:
            log.error("Request returned status code: %s", r.status_code)
            res["response"] = None
        return res
