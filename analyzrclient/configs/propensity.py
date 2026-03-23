"""Configuration dataclasses for propensity scoring workflows.

Provides ``PropensityTrainConfig`` and ``PropensityPredictConfig`` used to
parameterize propensity model training and prediction jobs submitted to the
analyzr analytics API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropensityTrainConfig:
    """Configuration for propensity model training.

    Specifies variable roles, algorithm selection, class-imbalance handling,
    cross-validation strategy, and runtime options used when fitting a
    propensity (classification) model on the analytics engine.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param outcome_var: Name of the binary target variable column, or
        ``None`` when the outcome is inferred from the dataset schema.
    :param categorical_vars: Names of categorical feature columns to include
        in the model.
    :param numerical_vars: Names of continuous/numerical feature columns to
        include in the model.
    :param bool_vars: Names of boolean feature columns to include in the
        model.
    :param algorithm: Classification algorithm identifier.  Defaults to
        ``'random-forest-classifier'``.
    :param train_size: Fraction of records used for training; the remainder
        forms the held-out test set.  Defaults to ``0.5``.
    :param smote: When ``True``, applies SMOTE oversampling to address class
        imbalance before fitting.
    :param param_grid: Hyper-parameter grid passed to the cross-validated
        grid-search during training, or ``None`` to use algorithm defaults.
    :param scoring: Scikit-learn scorer string used for grid-search
        evaluation (e.g. ``'roc_auc'``), or ``None`` to use the algorithm's
        default scorer.
    :param n_splits: Number of cross-validation folds, or ``None`` to use
        the analytics engine's default.
    :param out_of_core: When ``True``, enables out-of-core (incremental)
        training for datasets that exceed available memory.
    :param unique_categories: Exhaustive list of category labels across all
        categorical variables.  Required when the prediction dataset may
        contain categories not observed in training.
    """

    idx_var: str | None = None
    outcome_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = "random-forest-classifier"
    train_size: float = 0.5
    smote: bool = False
    param_grid: dict[str, Any] | None = None
    scoring: str | None = None
    n_splits: int | None = None
    out_of_core: bool = False
    unique_categories: list[str] = field(default_factory=list)


@dataclass
class PropensityPredictConfig:
    """Configuration for propensity model prediction.

    Specifies the variable sets and API batching options used when scoring
    new records with a previously trained propensity model.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param categorical_vars: Names of categorical feature columns expected by
        the trained model.
    :param numerical_vars: Names of continuous/numerical feature columns
        expected by the trained model.
    :param bool_vars: Names of boolean feature columns expected by the
        trained model.
    :param api_batch_size: Maximum number of records sent per API request.
        Defaults to ``2000``.
    """

    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    api_batch_size: int = 2000
