from __future__ import annotations

import json
import logging
import webbrowser
from datetime import datetime, timedelta
from typing import Any

import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ApiClient")

TOKEN_REFRESH: int = 10
MAX_ATTEMPTS: int = 3


class SamlSsoAuthClient:
    """Manages low-level interactions with the Analyzr API using SAML SSO authentication."""

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
        self._token = {'token': ''}
        self._token_time = None

    def get(self, uri: str) -> dict[str, Any]:
        self._check_token()
        r = requests.get(uri, params=self._token)
        return self._response(r)

    def post(self, uri: str, json_obj: Any, compressed: bool = False) -> dict[str, Any]:
        data = json.dumps(json_obj)
        self._check_token()
        r = requests.post(uri, json=data, params=self._token)
        return self._response(r)

    def _response(self, r: requests.Response) -> dict[str, Any]:
        res: dict[str, Any] = {'status': r.status_code}
        if 200 <= r.status_code < 300:
            res['response'] = r.json()
        else:
            log.error('WARNING! Request returned status code: %s', r.status_code)
            res['response'] = None
        return res

    def login(self, attempts: int = 0, verbose: bool = False) -> int:
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
        request = requests.get(f"https://{self.host}/saml/login-client/?client_id={self.client_id}")
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
        return bool(self._token.get('token'))

    def clean_token(self) -> None:
        self.url = None
        self.client_id = None
        self._token = {'token': ''}

    def _save_token(self, token: dict[str, str]) -> None:
        self._token = token
        self._token_time = datetime.now()

    def refresh_token(self) -> int:
        log.info("Starting: Refresh Token")
        request = requests.post(f"https://{self.host}/saml/refresh-token/", self._token)
        if request.status_code == 200:
            log.info("Success: Refreshing Token")
            self._save_token(request.json())
            return request.status_code
        log.error("Failed %s: Refresh Token Expired", request.status_code)
        return request.status_code

    def _check_token(self) -> None:
        if self._token_time and datetime.now() > self._token_time + timedelta(days=TOKEN_REFRESH):
            status = self.refresh_token()
            if status != 200:
                self.clean_token()
                self.login()
