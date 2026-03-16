from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ClusterTrainResult:
    """Result from cluster training."""
    model_id: str
    data: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    distances: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'request_id': self.model_id,
            'data': self.data,
            'stats': self.stats,
            'distances': self.distances,
        }


@dataclass
class ClusterPredictResult:
    """Result from cluster prediction."""
    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'data2': self.data,
        }
