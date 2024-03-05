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
# import bz2
import gzip
import logging
import webbrowser

from datetime import datetime, timedelta


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ApiClient")

TOKEN_REFRESH = 10  # every 10 days refresh jwt token
MAX_ATTEMPTS = 3 # number of login attempts by the client if a 408 response is received

class SamlSsoAuthClient:
    """
    The SamlSsoAuthClient class manages low-level interactions with the Analyzr API
    using SAML SSO authentication.

    """

    def __init__(self, host=None, verbose=False):
        """
        Initialize API client object

        :param host: FQDN for the API including schema, e.g. https://acme.api.g2m.ai
        :param verbose: Set to true for verbose output
        :return:
        """
        if verbose: log.info("Starting: Init Client")
        self.host = host
        self.url = None
        self.client_id = None
        self._token = {
            'token': ''
        }
        self._token_time = None
        return

    ############################################################################
    #                           Basic API calls                                #
    ############################################################################

    def _get(self, uri):
        """
        Returns standard GET response for URI. Note that this
        function is protected, not private.

        :param uri:
        :return res:
        """
        self._check_token()
        r = requests.get(uri, params=self._token)
        return self._response(r)

    def _post(self, uri, json_obj, compressed=False):
        """
        Returns standard POST response for URI. Note that this
        function is protected, not private.

        :param uri:
        :param json_obj:
        :param compressed:
        :return res:
        """
        data = json.dumps(json_obj)
        self._check_token()
        if compressed is False:
            r = requests.post(
                uri,
                json=data,
                params=self._token,
            )
        else:
            # Deprecated /compressed endpoint
            r = requests.post(
                uri,
                json=data,
                params=self._token,
            )
        return self._response(r)

    def _response(self, r):
        """
        """
        res = {}
        res['status'] = r.status_code
        if r.status_code>=200 and r.status_code<300:
            res['response'] = r.json()
        else:
            log.error('WARNING! Request returned status code: {}'.format(r.status_code))
            res['response'] = None
        return res


    ############################################################################
    #                      Authentication methods                              #
    ############################################################################

    def _login(self, attempts=0, verbose=False):
        """
        The client calls the API at <host> and handles the SAML SSO flow

        :param attempts: used to limit max muber of attempts when current login fails
        :param verbose: Set to true for verbose output
        :return status_code:
        """
        if verbose: log.info("Starting: Login")

        # Get URL for Client
        if not self.url:
            if verbose: log.info("[1] Retrieving URL...")
            uri = f"https://{self.host}/saml/login-client/?get_url=True"
            request = requests.get(uri)
            if request.status_code==200:
                if verbose: log.info("[2] Success: Get URL")
                data = request.json()
                self.url = data.get("url")
                self.client_id = data.get("client_id")
            else:
                log.error(f"Failed {request.status_code}: Get URL")
                return request.status_code

        # Get Token after login
        webbrowser.open(self.url)
        if verbose: log.info("[5] Requesting token...")
        request = requests.get(f"https://{self.host}/saml/login-client/?client_id={self.client_id}")
        status_code = request.status_code

        if status_code==200:
            if verbose: log.info("Success: Login")
            self._save_token(request.json())
            return status_code
        elif status_code==408:
            if attempts<MAX_ATTEMPTS-1:
                if verbose: log.info("Trying again... [{}/{}]".format(attempts+1, MAX_ATTEMPTS-1))
                status_code = self._login(attempts=attempts+1, verbose=verbose)
            else:
                log.error(f"Failed {status_code}: Login")

        return status_code

    def _logout(self, verbose=False):
        """
        Log the client out

        :param verbose: Set to true for verbose output
        :return:
        """
        if verbose: log.info("Starting: Logout")
        data = self._get(f"https://{self.host}/saml/logout-client/") # Logout
        # if verbose: log.info("data: {}".format(data))
        resp = data.get("response")
        if resp:
            logout_url = resp.get("url")
            if logout_url: webbrowser.open(logout_url)
        self.clean_token() # Clean up Token
        return


    ############################################################################
    #                      Token-related methods                               #
    ############################################################################

    @property
    def is_token(self):
        return bool(self._token.get('token'))

    def clean_token(self):
        self.url = None
        self.client_id = None
        self._token = {
            'token': ''
        }

    def _save_token(self, token):
        # Set token for the future requests
        self._token = token
        self._token_time = datetime.now()

    def refresh_token(self):
        log.info("Starting: Refresh Token")

        # Refresh token
        request = requests.post(f"https://{self.host}/saml/refresh-token/", self._token)

        if request.status_code == 200:
            log.info("Success: Refreshing Token")
            self._save_token(request.json())
            return request.status_code

        log.error(f"Failed {request.status_code}: Refresh Token Expired")
        return request.status_code

    def _check_token(self):
        """
        Check if token is still valid, re-login into system if refresh token is expired

        :return:
        """
        if self._token_time and datetime.now() > self._token_time + timedelta(days=TOKEN_REFRESH):
            status = self.refresh_token()
            if status != 200:
                self.clean_token()
                self.login()
        return
