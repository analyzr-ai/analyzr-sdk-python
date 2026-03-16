from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropensityTrainConfig:
    """Configuration for propensity training."""
    idx_var: str | None = None
    outcome_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = 'random-forest-classifier'
    train_size: float = 0.5
    smote: bool = False
    param_grid: dict[str, Any] | None = None
    scoring: str | None = None
    n_splits: int | None = None
    out_of_core: bool = False
    unique_categories: list[str] = field(default_factory=list)


@dataclass
class PropensityPredictConfig:
    """Configuration for propensity prediction."""
    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    api_batch_size: int = 2000
