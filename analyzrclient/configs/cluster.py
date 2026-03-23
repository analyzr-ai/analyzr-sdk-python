"""Configuration dataclasses for clustering workflows.

Provides ``ClusterTrainConfig`` and ``ClusterPredictConfig`` used to
parameterize cluster model training and prediction jobs submitted to the
analyzr analytics API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterTrainConfig:
    """Configuration for cluster model training.

    Specifies the variable sets, algorithm, and runtime options used when
    fitting a clustering model on the analytics engine.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param categorical_vars: Names of categorical feature columns to include
        in the model.
    :param numerical_vars: Names of continuous/numerical feature columns to
        include in the model.
    :param bool_vars: Names of boolean feature columns to include in the
        model.
    :param algorithm: Clustering algorithm identifier.  Defaults to
        ``'pca-kmeans'``.
    :param n_components: Number of PCA components (or cluster count,
        depending on algorithm) to use during fitting.  Defaults to ``5``.
    :param cluster_batch_size: Number of records per mini-batch for
        incremental/out-of-core fitting, or ``None`` to use the full dataset
        in a single pass.
    :param out_of_core: When ``True``, enables out-of-core (incremental)
        training to handle datasets that exceed available memory.
    """

    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = "pca-kmeans"
    n_components: int = 5
    cluster_batch_size: int | None = None
    out_of_core: bool = False


@dataclass
class ClusterPredictConfig:
    """Configuration for cluster model prediction.

    Specifies the variable sets and batching options used when assigning
    cluster labels to new records using a previously trained clustering model.

    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param categorical_vars: Names of categorical feature columns expected by
        the trained model.
    :param numerical_vars: Names of continuous/numerical feature columns
        expected by the trained model.
    :param bool_vars: Names of boolean feature columns expected by the
        trained model.
    :param cluster_batch_size: Number of records per API request batch, or
        ``None`` to send all records in a single request.
    """

    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    cluster_batch_size: int | None = None
