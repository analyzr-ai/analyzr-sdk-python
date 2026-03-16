from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RegressionTrainResult:
    """Result from regression training."""
    model_id: str
    features: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    coefs: pd.DataFrame | None = None
    laggingsats: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'features': self.features,
            'stats': self.stats,
            'coefs': self.coefs,
            'laggingsats': self.laggingsats,
        }


@dataclass
class RegressionPredictResult:
    """Result from regression prediction."""
    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'data2': self.data,
        }
