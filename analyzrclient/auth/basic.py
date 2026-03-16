from __future__ import annotations

import json
from typing import Any

import requests


class BasicAuthClient:
    """Manages low-level interactions with the Analyzr API using basic authentication."""

    def __init__(self, user: str = '', pwd: str = '', host: str | None = None) -> None:
        if host is None:
            print('ERROR! Please provide a valid host, e.g. host=acme.api.g2m.ai')
            exit(1)
        self._user: str = user
        self._pwd: str = pwd
        self._base_url: str = f'https://{host}/api/v1'

    def get(self, uri: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None:
            r = requests.get(uri, auth=(self._user, self._pwd))
        else:
            r = requests.get(uri, auth=(self._user, self._pwd), params=params)
        return self._response(r)

    def post(self, uri: str, json_obj: Any, compressed: bool = False) -> dict[str, Any]:
        data = json.dumps(json_obj)
        r = requests.post(uri, auth=(self._user, self._pwd), json=data)
        return self._response(r)

    def _response(self, r: requests.Response) -> dict[str, Any]:
        res: dict[str, Any] = {'status': r.status_code}
        if 200 <= r.status_code < 300:
            res['response'] = r.json()
        else:
            print(f'WARNING! Request returned status code: {r.status_code}')
            res['response'] = None
        return res
