import os, sys, time, json
import pandas as pd
from copy import deepcopy
import datetime

from .client_saml_sso import SamlSsoAuthClient
from .constants import *
from .utils import *
from .runner_cluster import ClusterRunner
from .runner_propensity import PropensityRunner
from .runner_regression import RegressionRunner
from .runner_causal import CausalRunner
from .runner_performance import PerformanceRunner 
from .runner_task import TaskRunner


class Analyzer:
    """
    Parent class for Analyzr client. This is the class that should be instantiated
    by the client. For detailed methods, see the appropriate runner class.

    :param host: the FQDN for your API tenant
    :type host: str, required
    :verbose: Set to true for verbose output
    :type verbose: boolean, optional
    """

    def __init__(self, host=None, verbose=False):
        """
        """
        # General config
        self.__client = SamlSsoAuthClient(host=host, verbose=verbose)
        self.__base_url = 'https://{}/api/v1'.format(host)
        self.__uri = '{}/analytics/'.format(self.__base_url)
        self.test = TaskRunner(client=self.__client, base_url=self.__base_url)

        # Initialize specific endpoints
        self.cluster = ClusterRunner(client=self.__client, base_url=self.__base_url)
        self.propensity = PropensityRunner(client=self.__client, base_url=self.__base_url)
        self.regression = RegressionRunner(client=self.__client, base_url=self.__base_url)
        self.causal = CausalRunner(client=self.__client, base_url=self.__base_url)
        self.performance = PerformanceRunner(client=self.__client, base_url=self.__base_url)
        return

    def version(self):
        """
        Provide version info

        :return JSON object with version info and other metadata
        """
        copy_client = self.client_version()
        copy_api = self.api_version()
        return {
            'api': {
                'status': copy_api['status'], 
                'version': copy_api['response']['version'] if copy_api['status']==200 else 'N/A',
                'tenant': copy_api['response']['tenant'] if copy_api['status']==200 else 'N/A',
            }, 
            'client': {
                'version': copy_client['version'],
            }, 
            'copyright': copy_client['copyright'], 
        }

    def api_version(self):
        """
        Provide API version info

        :return: JSON object with API version and other metadata
        """
        return self.__client._post(self.__uri, {'command': 'version'})

    def client_version(self):
        """
        Provide client version info

        :return: JSON object with API version and other metadata
        """
        return ({'version': f'{CLIENT_VERSION}', 'copyright': f'{datetime.date.today().year} (c) Go2Market Insights Inc. All rights reserved. Patent pending. '})

    def login(self, verbose=False):
        """
        Log in to Analyzr API

        :param verbose: Set to true for verbose output Set to True for verbose screen output
        :type verbose: boolean, optional
        :rtype: None
        """
        status_code = self.__client._login(verbose=verbose)
        if status_code==200:
            print('Login successful')
        else:
            print('Could not log in (status code: {})'.format(status_code))

        return

    def logout(self, verbose=False):
        """
        Log out of Analyzr API

        :param verbose: Set to true for verbose output Set to True for verbose screen output
        :type verbose: boolean, optional
        :rtype: None
        """
        return self.__client._logout(verbose=verbose)
