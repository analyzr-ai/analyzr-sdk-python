"""SAML SSO authentication client for the Analyzr API.

Initiates browser-based SAML login, caches the resulting bearer token, and
automatically refreshes it when it approaches expiry.  All API interactions
(GET/POST) route through this client so token management is transparent to callers.
"""

from __future__ import annotations

import json
import logging
import webbrowser
from datetime import datetime, timedelta
from typing import Any

import requests

log = logging.getLogger(__name__)

TOKEN_REFRESH: int = 10
MAX_ATTEMPTS: int = 3


class SamlSsoAuthClient:
    """Manages low-level interactions with the Analyzr API using SAML SSO authentication.

    :param host: API hostname without protocol (e.g. ``acme.api.g2m.ai``).
    :param verbose: Emit informational log messages during initialization when ``True``.
    """

    host: str | None
    url: str | None
    client_id: str | None
    _token: dict[str, str]
    _token_time: datetime | None

    def __init__(self, host: str | None = None, verbose: bool = False) -> None:
        if verbose:
            log.info("Starting: Init Client")
        self.host = host
        self.url = None
        self.client_id = None
        self._token = {"token": ""}
        self._token_time = None

    def get(self, uri: str) -> dict[str, Any]:
        """Send an authenticated GET request to the given URI.

        :param uri: Fully qualified URL to request.
        :return: Normalized response dict with ``status`` and ``response`` keys.
        :rtype: dict
        """
        self._check_token()
        r = requests.get(uri, params=self._token)
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
        self._check_token()
        r = requests.post(uri, json=data, params=self._token)
        return self._response(r)

    def _response(self, r: requests.Response) -> dict[str, Any]:
        res: dict[str, Any] = {"status": r.status_code}
        if 200 <= r.status_code < 300:
            res["response"] = r.json()
        else:
            log.error("WARNING! Request returned status code: %s", r.status_code)
            res["response"] = None
        return res

    def login(self, attempts: int = 0, verbose: bool = False) -> int:
        """Initiate SAML SSO login, opening the IdP URL in the default browser.

        Retries up to ``MAX_ATTEMPTS`` times on HTTP 408 (request timeout from
        the identity provider).  On success the token is saved internally and
        subsequent API calls proceed without re-authentication.

        :param attempts: Current retry count; callers should leave this at the default.
        :param verbose: Emit step-by-step log messages when ``True``.
        :return: HTTP status code of the final login request.
        :rtype: int
        """
        if verbose:
            log.info("Starting: Login")
        if not self.url:
            if verbose:
                log.info("[1] Retrieving URL...")
            uri = f"https://{self.host}/saml/login-client/?get_url=True"
            request = requests.get(uri)
            if request.status_code == 200:
                if verbose:
                    log.info("[2] Success: Get URL")
                data = request.json()
                self.url = data.get("url")
                self.client_id = data.get("client_id")
            else:
                log.error("Failed %s: Get URL", request.status_code)
                return request.status_code

        if self.url:
            webbrowser.open(self.url)
        if verbose:
            log.info("[5] Requesting token...")
        request = requests.get(
            f"https://{self.host}/saml/login-client/?client_id={self.client_id}"
        )
        status_code = request.status_code

        if status_code == 200:
            if verbose:
                log.info("Success: Login")
            self._save_token(request.json())
            return status_code
        elif status_code == 408:
            if attempts < MAX_ATTEMPTS - 1:
                if verbose:
                    log.info("Trying again... [%s/%s]", attempts + 1, MAX_ATTEMPTS - 1)
                status_code = self.login(attempts=attempts + 1, verbose=verbose)
            else:
                log.error("Failed %s: Login", status_code)

        return status_code

    def logout(self, verbose: bool = False) -> None:
        """Log out of the SAML session and clear the cached token.

        Opens the IdP logout URL in the default browser when provided by the server.

        :param verbose: Emit a log message at the start of the logout flow when ``True``.
        """
        if verbose:
            log.info("Starting: Logout")
        data = self.get(f"https://{self.host}/saml/logout-client/")
        resp = data.get("response")
        if resp:
            logout_url = resp.get("url")
            if logout_url:
                webbrowser.open(logout_url)
        self.clean_token()

    @property
    def is_token(self) -> bool:
        """Return ``True`` when a non-empty bearer token is currently cached.

        :rtype: bool
        """
        return bool(self._token.get("token"))

    def clean_token(self) -> None:
        """Clear the cached token, login URL, and client ID.

        Called after logout or when a token refresh fails and a full re-login is needed.
        """
        self.url = None
        self.client_id = None
        self._token = {"token": ""}

    def _save_token(self, token: dict[str, str]) -> None:
        self._token = token
        self._token_time = datetime.now()

    def refresh_token(self) -> int:
        """Refresh the bearer token using the server's refresh endpoint.

        :return: HTTP status code; ``200`` on success, non-200 on failure.
        :rtype: int
        """
        log.info("Starting: Refresh Token")
        request = requests.post(f"https://{self.host}/saml/refresh-token/", self._token)
        if request.status_code == 200:
            log.info("Success: Refreshing Token")
            self._save_token(request.json())
            return request.status_code
        log.error("Failed %s: Refresh Token Expired", request.status_code)
        return request.status_code

    def _check_token(self) -> None:
        if self._token_time and datetime.now() > self._token_time + timedelta(
            days=TOKEN_REFRESH
        ):
            status = self.refresh_token()
            if status != 200:
                self.clean_token()
                self.login()
