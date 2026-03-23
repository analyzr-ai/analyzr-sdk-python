"""Configuration dataclasses for Media Mix Modeling (MMM) workflows.

Provides ``MMMTrainConfig`` and ``MMMOptimizeConfig`` used to parameterize
MMM model training and budget optimization jobs submitted to the analyzr
analytics API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MMMTrainConfig:
    """Configuration for MMM model training.

    Specifies the variable roles and algorithm used when fitting a Media Mix
    Model on the analytics engine.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param time_var: Name of the column representing the time dimension
        (e.g. a date or week identifier), or ``None`` when inferred.
    :param outcome_var: Name of the target/outcome variable column (e.g.
        revenue or sales), or ``None`` when inferred from the dataset schema.
    :param media_vars: Names of media spend or impression columns whose
        contribution to the outcome will be modelled.
    :param other_vars: Names of non-media covariate columns included as
        control variables.
    :param algorithm: MMM algorithm identifier.  Defaults to
        ``'mmm-carryover'``.
    """

    idx_var: str | None = None
    time_var: str | None = None
    outcome_var: str | None = None
    media_vars: list[str] = field(default_factory=list)
    other_vars: list[str] = field(default_factory=list)
    algorithm: str = "mmm-carryover"


@dataclass
class MMMOptimizeConfig:
    """Configuration for MMM budget optimization.

    Specifies the total budget constraint used when running a spend
    optimization pass on a previously trained MMM model.

    :param budget: Total budget to allocate across media channels, or
        ``None`` to use the historical total spend observed in the training
        data.
    """

    budget: float | None = None
