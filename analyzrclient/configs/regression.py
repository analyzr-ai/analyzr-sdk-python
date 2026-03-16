from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REGRESSION_DEFAULT_ALGO: str = 'linear-regression'

LINEAR_ALGORITHMS: list[str] = [
    'linear-regression',
    'bayesian-ridge-regression',
    'lasso-regression',
    'ridge-regression',
]


@dataclass
class RegressionTrainConfig:
    """Configuration for regression training."""
    idx_var: str | None = None
    outcome_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    saturation_vars: list[str] = field(default_factory=list)
    lagging_vars: list[str] = field(default_factory=list)
    algorithm: str = REGRESSION_DEFAULT_ALGO
    train_size: float = 0.5
    param_grid: dict[str, Any] | None = None


@dataclass
class RegressionPredictConfig:
    """Configuration for regression prediction."""
    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
