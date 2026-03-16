from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PropensityTrainResult:
    """Result from propensity training."""
    model_id: str
    features: pd.DataFrame | None = None
    confusion_matrix: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    roc: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'features': self.features,
            'confusion_matrix': self.confusion_matrix,
            'stats': self.stats,
            'roc': self.roc,
        }


@dataclass
class PropensityPredictResult:
    """Result from propensity prediction."""
    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'data2': self.data,
        }
