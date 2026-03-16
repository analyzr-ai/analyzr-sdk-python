# Python SDK for the Analyzr API

## Overview

This Python client provides access to the G2M Analyzr API for running ML analytics workflows including clustering, propensity scoring, regression, causal analysis, marketing mix modeling, and performance analysis.

- General information: https://analyzr.ai
- Help and support: https://help.analyzr.ai
- SDK reference documentation: https://analyzr-sdk-python.readthedocs.io

## Installation

### For users

```bash
pip install analyzr
```

### For developers

1. Clone the repository:
```bash
git clone https://github.com/analyzr-ai/analyzr-sdk-python.git
cd analyzr-sdk-python
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install the package in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

This installs the SDK locally so changes to source files are reflected immediately without reinstalling.

## Quick start

```python
from analyzrclient import Analyzer

analyzer = Analyzer(host="<your-tenant>.analyzr.ai")
analyzer.login()
analyzer.version()
```

## Testing

Tests run against a live API tenant. Before running tests, update `tests/config.json` with your API host:

```json
{
  "host": "<your-tenant>.analyzr.ai"
}
```

### Running tests

```bash
# All quick tests
pytest tests/test_quick.py

# Full test suite
pytest tests/test_all.py

# Verbose output (shows each test name and PASSED/FAILED)
pytest tests/test_quick.py -v

# Show print statements and log output (disables stdout capture)
pytest tests/test_quick.py -v -s

# Run a specific test class
pytest tests/test_quick.py::ClusteringTest -v

# Run a single test method
pytest tests/test_quick.py::ClusteringTest::test_birch -v

# Exclude a test class (useful for skipping slow or WIP tests)
pytest tests/test_quick.py -v -k "not Performance"

# Run only tests matching a keyword
pytest tests/test_quick.py -v -k "propensity"

# Stop on first failure
pytest tests/test_quick.py -v -x

# Show local variables in tracebacks
pytest tests/test_quick.py -v --tb=long

# Run with short traceback (just the assertion)
pytest tests/test_quick.py -v --tb=short
```

### Test files

| File | Description |
|------|-------------|
| `tests/test_quick.py` | One test per analytics domain — fast validation |
| `tests/test_all.py` | Full coverage with multiple algorithms per domain |
| `tests/utils.py` | Shared dataset loaders (Titanic, banking, causal, MMM, performance) |
| `tests/config.json` | API tenant configuration (not committed) |

### Type checking

```bash
pyright
```

### Linting

```bash
ruff check .
ruff format .
```
