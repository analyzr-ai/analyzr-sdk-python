"""Result dataclasses for propensity scoring workflows.

Provides ``PropensityTrainResult`` and ``PropensityPredictResult`` which
encapsulate the structured outputs returned by the analyzr analytics API
after propensity model training and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PropensityTrainResult:
    """Result from propensity model training.

    Holds the model identifier and the DataFrames produced by the analytics
    engine after fitting a propensity (classification) model.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent prediction requests.
    :param features: DataFrame of feature importances ranked by their
        contribution to the classifier, or ``None`` if not returned.
    :param confusion_matrix: DataFrame representation of the confusion matrix
        evaluated on the held-out test set, or ``None`` if not returned.
    :param stats: DataFrame of classification performance metrics (e.g.
        accuracy, precision, recall, F1, AUC), or ``None`` if not returned.
    :param roc: DataFrame of ROC curve points (FPR/TPR pairs) for plotting
        or threshold selection, or ``None`` if not returned.
    """

    model_id: str
    features: pd.DataFrame | None = None
    confusion_matrix: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    roc: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id``, ``features``,
            ``confusion_matrix``, ``stats``, and ``roc``.
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "features": self.features,
            "confusion_matrix": self.confusion_matrix,
            "stats": self.stats,
            "roc": self.roc,
        }


@dataclass
class PropensityPredictResult:
    """Result from propensity model prediction.

    Holds the model identifier and the DataFrame of scored records produced
    by inference against a trained propensity model.

    :param model_id: Unique identifier of the model used for prediction.
    :param data: DataFrame of input records annotated with predicted
        propensity scores (probabilities), or ``None`` if not returned by
        the API.
    """

    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id`` and ``data2`` (the scored
            DataFrame, keyed as ``data2`` for API compatibility).
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "data2": self.data,
        }
