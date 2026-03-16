# Development Guide

## Prerequisites

- Python 3.9+
- Git Bash (Windows) or terminal (macOS/Linux)

## Environment Setup

Virtual environments are kept **outside** the repository to avoid `.gitignore` clutter.

### Create the virtual environment

```bash
# Windows (Git Bash)
python -m venv /c/Users/$USER/venvs/analyzr-sdk

# macOS / Linux
python -m venv ~/venvs/analyzr-sdk
```

### Activate

```bash
# Windows (Git Bash) — use full path, ~ does not expand reliably
source /c/Users/$USER/venvs/analyzr-sdk/Scripts/activate

# macOS / Linux
source ~/venvs/analyzr-sdk/bin/activate
```

### Install dependencies

```bash
pip install -e .
```

### Deactivate

```bash
deactivate
```

### Delete the environment

If you need to start fresh, deactivate first then remove the directory:

```bash
deactivate
rm -rf /c/Users/$USER/venvs/analyzr-sdk   # Windows (Git Bash)
rm -rf ~/venvs/analyzr-sdk                 # macOS / Linux
```

## Git Workflow

All changes go through feature branches and pull request review. Direct pushes to `main` are not allowed.

```bash
# 1. Create a feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 2. Make changes, commit
git add <files>
git commit -m "Description of change"

# 3. Push feature branch to remote
git push origin feature/your-feature-name

# 4. Open a pull request to main on GitHub
# 5. Get approval from at least one reviewer
# 6. Merge via GitHub (squash or merge commit per team preference)
```

## Building

Build the package locally to verify everything compiles correctly:

```bash
pip install build
python -m build
```

This creates `dist/` with a `.whl` and `.tar.gz`. Nothing is published — these are local artifacts only.

### Clean up build artifacts

```bash
rm -rf dist/ build/ *.egg-info analyzr_sdk_python.egg-info
```

## Publishing

### Automated (GitHub Actions)

Push a version tag to trigger the publish workflow:

```bash
# 1. Update version in pyproject.toml
# 2. Commit the change
# 3. Tag and push
git tag v2.0.0
git push origin main --tags
```

GitHub Actions builds the package and publishes to PyPI automatically via trusted publishing (OIDC).

Trusted publishing must be configured once on PyPI:
1. Go to https://pypi.org/manage/project/analyzr/settings/publishing/
2. Add a trusted publisher with owner `analyzr-ai`, repo `analyzr-sdk-python`, workflow `publish.yml`

### Manual (deprecated)

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Version Management

The package version is defined in one place: `pyproject.toml` under `[project] version`.

At runtime, `analyzrclient/constants.py` reads the installed version via `importlib.metadata.version()`. If the package isn't installed (e.g. running from source without `pip install -e .`), it falls back to `'dev'`.

## Project Structure

```
analyzrclient/
├── __init__.py              # Package entry point — exports Analyzer
├── analyzer.py              # Main Analyzer class (aggregates all runners)
├── client_saml_sso.py       # SAML SSO authentication client
├── constants.py             # Global constants (version, paths)
├── runner_base.py           # Base class for all runners (buffer, encoding, polling)
├── runner_performance.py    # Performance analysis runner
├── runner_regression.py     # Regression runner
├── runner_propensity.py     # Propensity runner
├── runner_cluster.py        # Clustering runner
├── runner_causal.py         # Causal analysis runner
├── runner_mmm.py            # Marketing Mix Modeling runner
├── runner_task.py           # Task runner (health checks)
└── utils.py                 # Encoding utilities & helpers
```
