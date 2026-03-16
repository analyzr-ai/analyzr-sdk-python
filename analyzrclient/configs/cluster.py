from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterTrainConfig:
    """Configuration for cluster training."""
    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    algorithm: str = 'pca-kmeans'
    n_components: int = 5
    cluster_batch_size: int | None = None
    out_of_core: bool = False


@dataclass
class ClusterPredictConfig:
    """Configuration for cluster prediction."""
    idx_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)
    numerical_vars: list[str] = field(default_factory=list)
    bool_vars: list[str] = field(default_factory=list)
    cluster_batch_size: int | None = None
