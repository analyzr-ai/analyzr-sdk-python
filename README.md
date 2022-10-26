# Python SDK for the Analyzr API

## Overview
This Python client will give you access to the Analyzr API. See files in the `examples` folder
for examples showing how to use the client. Note that a `client_id` should always be provided when querying the API; it is used for reporting purposes.
* For general information please see https://analyzr.ai.
* For help and support see https://support.analyzr.ai.
* For SDK reference documentation see  https://analyzr-sdk-python.readthedocs.io.

## Installation instructions
Getting the client set up will require the following:

1. Install the latest version of the client on your local machine:
```
pip install analyzr-sdk-python
```

2. Get an API username and password from your Analyzr admin (you may need SSO credentials from your local admin instead).

3. Confirm you are able to connect to the API, and check the API version
as follows from a Python session:
```
>>> from analyzrclient import Analyzer
>>> analyzer = Analyzer(host="<your host>")
>>> analyzer.login()
Login successful
>>> Analyzer().version()
{'status': 200, 'response': {'version': 'x.x.xxx', 'tenant': <your tenant name>, 'copyright': '2022 (c) Go2Market Insights LLC. All rights reserved.'}}
```
