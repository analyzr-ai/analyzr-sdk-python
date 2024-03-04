"""
Copyright (c) 2024 Go2Market Insights, Inc d/b/a Analyzr
All rights reserved.
https://analyzr.ai

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import requests
import json
from copy import deepcopy

from .constants import *


class BasicAuthClient:
    """
    The BasicAuthClient class manages low-level interactions with the Analyzr API
    using basic authentication (when applicable).

    """
    def __init__(self, user=USER, pwd=PWD, host=None):
        """
        :param user:
        :param pwd:
        :param host:
        :return:
        """
        if host is None:
            print('ERROR! Please provide a valid host, e.g. host=acme.api.g2m.ai')
            exit(1)
        self.__user = user
        self.__pwd = pwd
        self._base_url = 'https://{}/api/v1'.format(host)
        return

    ############################################################################
    #                           Basic API calls                                #
    ############################################################################

    def _get(self, uri, params=None):
        """
        Returns standard GET response for URI. Note that this
        function is protected, not private.
        :param uri:
        :param params:
        :return res:
        """
        if params is None:
            r = requests.get(uri, auth=(self.__user, self.__pwd))
        else:
            r = requests.get(uri, auth=(self.__user, self.__pwd), params=params)
        return self._response(r)

    def _post(self, uri, json_obj, compressed=False):
        """
        Returns standard POST response for URI. Note that this
        function is protected, not private.
        :param uri:
        :param json_obj:
        :param compressed: not used at this time
        :return res:
        """
        data = json.dumps(json_obj)
        r = requests.post(uri, auth=(self.__user, self.__pwd), json=data)
        return self._response(r)

    def _response(self, r):
        """
        """
        res = {}
        res['status'] = r.status_code
        if r.status_code>=200 and r.status_code<300:
            res['response'] = r.json()
        else:
            print('WARNING! Request returned status code: {}'.format(r.status_code))
            res['response'] = None
        return res
