"""Configuration dataclasses for causal inference workflows.

Provides ``CausalTrainConfig`` used to parameterize causal analysis
(treatment-effect estimation) training jobs submitted to the analyzr
analytics API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalTrainConfig:
    """Configuration for causal analysis model training.

    Specifies variable roles, the causal estimation algorithm, and optional
    standard-error computation used when fitting a causal inference model on
    the analytics engine.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param outcome_var: Name of the outcome variable column whose
        treatment effect is being estimated, or ``None`` when inferred from
        the dataset schema.
    :param treatment_var: Name of the binary treatment indicator column, or
        ``None`` when inferred from the dataset schema.
    :param categorical_vars: Names of categorical covariate columns to
        include in the model.
    :param numerical_vars: Names of continuous/numerical covariate columns
        to include in the model.
    :param bool_vars: Names of boolean covariate columns to include in the
        model.
    :param algorithm: Causal estimation algorithm identifier.  Defaults to
        ``'propensity-score-matching-ci'``.
    :param standard_error: When ``True``, computes bootstrap standard errors
        for the estimated treatment effect.  Increases computation time.
    """

    idx_var: str | None = None
    outcome_var: str | None = None
    treatment_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = "propensity-score-matching-ci"
    standard_error: bool = False
