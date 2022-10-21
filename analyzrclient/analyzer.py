"""
Copyright (c) 2020-2022 Go2Market Insights, LLC d/b/a Analyzr
All rights reserved.
https://analyzr.ai

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .client_saml_sso import SamlSsoAuthClient
from .constants import *
from .utils import *
from .runner_cluster import ClusterRunner
from .runner_propensity import PropensityRunner
from .runner_regression import RegressionRunner
from .runner_task import TaskRunner


class Analyzer:
    """
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
        return

    def version(self):
        """
        Returns API version info

        :return version:
        """
        return self.__client._post(self.__uri, {'command': 'version'})

    def login(self, verbose=False):
        """
        :param verbose:
        :return status_code:
        """
        status_code = self.__client._login(verbose=verbose)
        if status_code==200:
            print('Login successful')
        else:
            print('Could not log in (status code: {})'.format(status_code))

        return

    def logout(self, verbose=False):
        """
        :param verbose:
        :return:
        """
        return self.__client._logout(verbose=verbose)

    def export(self, model_id, verbose=False):
        """
        :param model_id:
        :param verbose:
        :return res:
        """
        return self.__client._post(
            self.__uri,
            {
                'command': 'model-export',
                'model_id': model_id,
            }
        )
