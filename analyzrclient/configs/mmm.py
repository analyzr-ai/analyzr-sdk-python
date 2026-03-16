from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MMMTrainConfig:
    """Configuration for MMM training."""
    idx_var: str | None = None
    time_var: str | None = None
    outcome_var: str | None = None
    media_vars: list[str] = field(default_factory=list)
    other_vars: list[str] = field(default_factory=list)
    algorithm: str = 'mmm-carryover'


@dataclass
class MMMOptimizeConfig:
    """Configuration for MMM budget optimization."""
    budget: float | None = None
