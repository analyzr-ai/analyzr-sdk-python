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

To check your tenant and version info:

>>> Analyzer().version()
{'status': 200, 'response': { 'version': 'x.x.xxx', 'tenant': <your tenant name>, 'copyright': '2023 (c) Go2Market Insights Inc. All rights reserved. Patent pending.'}}

For example notebooks see the Analyzr Python SDK `repo <https://github.com/analyzr-ai/analyzr-sdk-python/>`_.
