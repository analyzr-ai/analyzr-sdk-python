# Python SDK for the Analyzr API

## Overview
This Python client will give you access to the G2M Analyzer API from any machine. See files in the `examples` folder
for examples showing how to use the client. Note that a `client_id` should always be provided when querying the API; it is used for reporting purposes.

## Installing the client in production mode:
Getting the client set up will require the following:

1. Install the latest version of the client on your local machine:
```
pip install analyzr-sdk-python
```
If it simply needs to be updated, use:
```
pip install analyzr-sdk-python --upgrade
```

2. Get an API username and password from your Analyzr admin (you may need SSO credentials from your local admin instead).

3. Make sure your IP is on the access list; the API will only accept calls from a valid IP on the access list. If you are on a client network, contact your G2M admin to update firewall rules.

4. To confirm you are able to connect to the API, make sure the `g2mclient` folder is your Python path and check the API version
as follows from a Python session:
```
>>> from analyzrclient import Analyzer
>>> analyzer = Analyzer(host="<your host>")
>>> analyzer.login()
Login successful
>>> Analyzer().version()
{'status': 200, 'response': {'version': 'x.x.xxx', 'tenant': <your tenant name>, 'copyright': '2022 (c) Go2Market Insights LLC. All rights reserved.'}}
```
