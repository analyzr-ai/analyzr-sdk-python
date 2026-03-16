from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MMMTrainResult:
    """Result from MMM training."""
    model_id: str
    train_data: pd.DataFrame | None = None
    train_stats: pd.DataFrame | None = None
    test_stats: pd.DataFrame | None = None
    lag_stats: pd.DataFrame | None = None
    lag_hist: pd.DataFrame | None = None
    media_spend: pd.DataFrame | None = None
    contrib_stats: pd.DataFrame | None = None
    contrib_data: pd.DataFrame | None = None
    resp_curve_media: pd.DataFrame | None = None
    resp_curve_pred: pd.DataFrame | None = None


@dataclass
class MMMOptimizeResult:
    """Result from MMM budget optimization."""
    model_id: str
    data: pd.DataFrame | None = None
