"""Configuration dataclasses for regression workflows.

Provides ``RegressionTrainConfig`` and ``RegressionPredictConfig`` used to
parameterize regression model training and prediction jobs submitted to the
analyzr analytics API.

Module-level constants ``REGRESSION_DEFAULT_ALGO`` and ``LINEAR_ALGORITHMS``
enumerate the supported linear algorithm identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REGRESSION_DEFAULT_ALGO: str = "linear-regression"

LINEAR_ALGORITHMS: list[str] = [
    "linear-regression",
    "bayesian-ridge-regression",
    "lasso-regression",
    "ridge-regression",
]


@dataclass
class RegressionTrainConfig:
    """Configuration for regression model training.

    Specifies the variable roles, algorithm selection, train/test split
    ratio, and optional hyper-parameter grid used when fitting a regression
    model on the analytics engine.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param outcome_var: Name of the target/outcome variable column, or
        ``None`` when the outcome is inferred from the dataset schema.
    :param categorical_vars: Names of categorical feature columns to include
        in the model.
    :param numerical_vars: Names of continuous/numerical feature columns to
        include in the model.
    :param bool_vars: Names of boolean feature columns to include in the
        model.
    :param saturation_vars: Names of variables to which a saturation
        (diminishing-returns) transform is applied before fitting.
    :param lagging_vars: Names of variables for which lagged versions are
        generated and added as features.
    :param algorithm: Regression algorithm identifier.  Must be one of
        ``LINEAR_ALGORITHMS``.  Defaults to ``REGRESSION_DEFAULT_ALGO``.
    :param train_size: Fraction of records used for training; the remainder
        forms the held-out test set.  Defaults to ``0.5``.
    :param param_grid: Hyper-parameter grid passed to the cross-validated
        grid-search during training, or ``None`` to use algorithm defaults.
    """

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
    """Configuration for regression model prediction.

    Specifies the variable sets used when generating predictions from a
    previously trained regression model.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param categorical_vars: Names of categorical feature columns expected by
        the trained model.
    :param numerical_vars: Names of continuous/numerical feature columns
        expected by the trained model.
    :param bool_vars: Names of boolean feature columns expected by the
        trained model.
    """

    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
