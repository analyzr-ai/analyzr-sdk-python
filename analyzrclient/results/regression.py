"""Result dataclasses for regression workflows.

Provides ``RegressionTrainResult`` and ``RegressionPredictResult`` which
encapsulate the structured outputs returned by the analyzr analytics API
after regression model training and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RegressionTrainResult:
    """Result from regression model training.

    Holds the model identifier and the DataFrames produced by the analytics
    engine after fitting a regression model.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent prediction requests.
    :param features: DataFrame of feature importances or selection results,
        or ``None`` if not returned by the API.
    :param stats: DataFrame of model performance statistics (e.g. R², RMSE,
        MAE across train/test splits), or ``None`` if not returned.
    :param coefs: DataFrame of fitted model coefficients with confidence
        intervals, or ``None`` if not returned.
    :param laggingsats: DataFrame of statistics for lagged variable
        transformations applied during training, or ``None`` if not returned.
    """

    model_id: str
    features: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    coefs: pd.DataFrame | None = None
    laggingsats: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id``, ``features``, ``stats``,
            ``coefs``, and ``laggingsats``.
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "features": self.features,
            "stats": self.stats,
            "coefs": self.coefs,
            "laggingsats": self.laggingsats,
        }


@dataclass
class RegressionPredictResult:
    """Result from regression model prediction.

    Holds the model identifier and the DataFrame of predicted values produced
    by inference against a trained regression model.

    :param model_id: Unique identifier of the model used for prediction.
    :param data: DataFrame of input records annotated with predicted outcome
        values, or ``None`` if not returned by the API.
    """

    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id`` and ``data2`` (the prediction
            DataFrame, keyed as ``data2`` for API compatibility).
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "data2": self.data,
        }
