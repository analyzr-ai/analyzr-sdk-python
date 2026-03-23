"""Result dataclasses for Media Mix Modeling (MMM) workflows.

Provides ``MMMTrainResult`` and ``MMMOptimizeResult`` which encapsulate the
structured outputs returned by the analyzr analytics API after MMM model
training and budget optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MMMTrainResult:
    """Result from MMM model training.

    Holds the model identifier and the DataFrames produced by the analytics
    engine after fitting a Media Mix Model.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent optimization requests.
    :param train_data: DataFrame of training-period actuals and model-fitted
        values, or ``None`` if not returned by the API.
    :param train_stats: DataFrame of in-sample performance statistics (e.g.
        R², MAPE) for the training period, or ``None`` if not returned.
    :param test_stats: DataFrame of out-of-sample performance statistics for
        the held-out test period, or ``None`` if not returned.
    :param lag_stats: DataFrame of carryover/lag parameter estimates per
        media channel, or ``None`` if not returned.
    :param lag_hist: DataFrame of lag weight distributions used for
        visualization or diagnostics, or ``None`` if not returned.
    :param media_spend: DataFrame of observed media spend per channel over
        the modelled period, or ``None`` if not returned.
    :param contrib_stats: DataFrame of aggregate contribution statistics
        summarizing each channel's share of the outcome, or ``None`` if not
        returned.
    :param contrib_data: DataFrame of time-series contribution decomposition
        showing each channel's contribution per period, or ``None`` if not
        returned.
    :param resp_curve_media: DataFrame of media-spend values used to
        construct response curves per channel, or ``None`` if not returned.
    :param resp_curve_pred: DataFrame of predicted outcome values
        corresponding to ``resp_curve_media`` points, or ``None`` if not
        returned.
    """

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
    """Result from MMM budget optimization.

    Holds the model identifier and the DataFrame of optimized budget
    allocations produced by the analytics engine.

    :param model_id: Unique identifier of the model against which
        optimization was run.
    :param data: DataFrame of recommended spend allocations per media channel
        with projected outcome lift, or ``None`` if not returned by the API.
    """

    model_id: str
    data: pd.DataFrame | None = None
