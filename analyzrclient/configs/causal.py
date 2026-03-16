from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalTrainConfig:
    """Configuration for causal analysis training."""
    idx_var: str | None = None
    outcome_var: str | None = None
    treatment_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = 'propensity-score-matching-ci'
    standard_error: bool = False
