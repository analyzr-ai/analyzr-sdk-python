from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CausalTrainResult:
    """Result from causal analysis training."""
    model_id: str
    atx: pd.DataFrame | None = None
    raw: pd.DataFrame | None = None
    misc: pd.DataFrame | None = None
    bins: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_id': self.model_id,
            'atx': self.atx,
            'raw': self.raw,
            'misc': self.misc,
            'bins': self.bins,
        }
