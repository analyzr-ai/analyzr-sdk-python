Usage
=====

.. _installation:

Installation
------------

To use the Analyzr Python SDK, first install it using pip:

.. code-block:: console

   (.venv) $ pip install analyzr-sdk-python

Connecting to the API
---------------------

To connect to the Analyzr API do as follows:

>>> from analyzrclient import Analyzer
>>> analyzer = Analyzer(host="<your host>")
>>> analyzer.login()
Login successful
>>> Analyzer().version()
{'status': 200, 'response': {'version': 'x.x.xxx', 'tenant': <your tenant name>, 'copyright': '2022 (c) Go2Market Insights LLC. All rights reserved.'}}
