"""Result dataclasses for clustering workflows.

Provides ``ClusterTrainResult`` and ``ClusterPredictResult`` which
encapsulate the structured outputs returned by the analyzr analytics API
after cluster model training and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ClusterTrainResult:
    """Result from cluster model training.

    Holds the model identifier and the DataFrames produced by the analytics
    engine after fitting a clustering model.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent prediction requests.
    :param data: DataFrame of input records annotated with their assigned
        cluster labels, or ``None`` if not returned by the API.
    :param stats: DataFrame of per-cluster summary statistics (e.g. centroid
        coordinates, intra-cluster variance), or ``None`` if not returned.
    :param distances: DataFrame of record-to-centroid distances or pairwise
        distance metrics, or ``None`` if not returned.
    """

    model_id: str
    data: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    distances: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id``, ``request_id`` (aliased to
            ``model_id`` for API compatibility), ``data``, ``stats``, and
            ``distances``.
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "request_id": self.model_id,
            "data": self.data,
            "stats": self.stats,
            "distances": self.distances,
        }


@dataclass
class ClusterPredictResult:
    """Result from cluster model prediction.

    Holds the model identifier and the DataFrame of records annotated with
    cluster assignments produced by inference against a trained model.

    :param model_id: Unique identifier of the model used for prediction.
    :param data: DataFrame of input records annotated with their predicted
        cluster labels, or ``None`` if not returned by the API.
    """

    model_id: str
    data: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id`` and ``data2`` (the predicted
            data DataFrame, keyed as ``data2`` for API compatibility).
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "data2": self.data,
        }
