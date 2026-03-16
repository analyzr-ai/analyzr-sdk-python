from __future__ import annotations

import datetime
import logging
from typing import Any

from .auth.saml_sso import SamlSsoAuthClient
from .constants import CLIENT_VERSION
from .runners import (
    CausalRunner,
    ClusterRunner,
    MMMRunner,
    PerformanceRunner,
    PropensityRunner,
    RegressionRunner,
    TaskRunner,
)

log = logging.getLogger(__name__)


class Analyzer:
    """Parent class for Analyzr client.

    This is the class that should be instantiated by the client.
    For detailed methods, see the appropriate runner class.

    :param host: the FQDN for your API tenant
    :type host: str, required
    :param verbose: Set to true for verbose output
    :type verbose: bool, optional
    """

    _client: SamlSsoAuthClient
    _base_url: str
    _uri: str
    test: TaskRunner
    cluster: ClusterRunner
    propensity: PropensityRunner
    regression: RegressionRunner
    causal: CausalRunner
    mmm: MMMRunner
    performance: PerformanceRunner

    def __init__(self, host: str | None = None, verbose: bool = False) -> None:
        self._client = SamlSsoAuthClient(host=host, verbose=verbose)
        self._base_url = f'https://{host}/api/v1'
        self._uri = f'{self._base_url}/analytics/'
        self.test = TaskRunner(client=self._client, base_url=self._base_url)
        self.cluster = ClusterRunner(client=self._client, base_url=self._base_url)
        self.propensity = PropensityRunner(client=self._client, base_url=self._base_url)
        self.regression = RegressionRunner(client=self._client, base_url=self._base_url)
        self.causal = CausalRunner(client=self._client, base_url=self._base_url)
        self.mmm = MMMRunner(client=self._client, base_url=self._base_url)
        self.performance = PerformanceRunner(client=self._client, base_url=self._base_url)

    def version(self) -> dict[str, Any]:
        """Provide version info."""
        copy_client = self.client_version()
        copy_api = self.api_version()
        return {
            'api': {
                'status': copy_api['status'],
                'version': copy_api['response']['version'] if copy_api['status'] == 200 else 'N/A',
                'tenant': copy_api['response']['tenant'] if copy_api['status'] == 200 else 'N/A',
            },
            'client': {
                'version': copy_client['version'],
            },
            'copyright': copy_client['copyright'],
        }

    def api_version(self) -> dict[str, Any]:
        """Provide API version info."""
        return self._client.post(self._uri, {'command': 'version'})

    def client_version(self) -> dict[str, str]:
        """Provide client version info."""
        return {
            'version': f'{CLIENT_VERSION}',
            'copyright': f'{datetime.date.today().year} (c) Go2Market Insights Inc. All rights reserved. Patent pending. ',
        }

    def login(self, verbose: bool = False) -> None:
        """Log in to Analyzr API."""
        status_code = self._client.login(verbose=verbose)
        if status_code == 200:
            log.info('Login successful')
        else:
            log.warning('Could not log in (status code: %s)', status_code)

    def logout(self, verbose: bool = False) -> None:
        """Log out of Analyzr API."""
        self._client.logout(verbose=verbose)
