"""
Copyright (c) 2020-2021 Go2Market Insights, LLC
All rights reserved.
https://g2m.ai

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

from .runner_base import BaseRunner
from .constants import *
from .utils import *

class TaskRunner(BaseRunner):
    """
    Runs pipeline for generic tasks (asynchronous testing)

    """
    def __init__(self, client=None, base_url=None):
        """
        """
        super().__init__(client=client, base_url=base_url)
        self.__uri = '{}/analytics/'.format(self._base_url)
        return

    def run(self, type='simple', verbose=False, compressed=False):
        """
        Returns test task output

        :param type: can be 'simple' (default) or 'storage'
        :param verbose: Set to true for verbose output
        :param compressed:
        :return res:
        """
        request_id = self._get_request_id()
        return self.__run(df=get_test_data(), request_id=request_id, client_id='test', type=type, verbose=verbose, compressed=compressed)

    def __run(self, df=None, request_id=None, client_id=None, type='simple', verbose=False, compressed=False):
        """
        :param df:
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param type: can be 'simple' (default), 'storage', or 'bz2'
        :param verbose: Set to true for verbose output
        :param comnpressed:
        :return res:
        """
        client_id = 'test'
        payload = {'request_id': request_id, 'client_id': client_id, 'type': type, 'command': 'test-task', 'data': df.to_dict()}
        self._client._post(self.__uri, payload, compressed=compressed)
        self._poll(payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'}, timeout=60, step=1, verbose=verbose)
        res = self._buffer_read(request_id=request_id, client_id=client_id, verbose=verbose, raw=True)
        return res if res['status']!=200 else {
            'status': res['status'],
            'response': {
                'request_id': res['response']['request_id'],
                'client_id': res['response']['client_id'],
                'data': pd.DataFrame(res['response']['data']),
                'type': type,
            }
        }
