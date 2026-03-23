"""Codec for propensity scoring result decoding.

Decodes feature importance, confusion matrix, model statistics, and ROC curve
frames returned by the analytics engine after a propensity training run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...results.propensity import PropensityTrainResult
from ..domain import DomainCodec
from ..field_name import FieldNameEncoder


class PropensityCodec(DomainCodec):
    """Decodes propensity scoring results."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        """Return buffer frame names required to decode propensity train results.

        :param config: Propensity training configuration (unused; included for interface parity).
        :return: List of frame names: ``features``, ``confusion_matrix``, ``stats``, ``roc``.
        :rtype: list[str]
        """
        return ["features", "confusion_matrix", "stats", "roc"]

    def decode_train_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> PropensityTrainResult:
        """Decode propensity buffer frames into a ``PropensityTrainResult``.

        :param frames: Buffer frames: ``features``, ``confusion_matrix``, ``stats``, ``roc``.
        :param keys: Encoding keys; uses ``fref_exp`` for feature name decoding.
        :param config: Propensity training configuration.
        :param model_id: Unique identifier for the trained model.
        :param encoding: Decode feature field names when ``True``.
        :return: Decoded propensity train result.
        :rtype: PropensityTrainResult
        """
        fref = keys.get("fref_exp", {})

        features = _decode_features(
            frames.get("features", pd.DataFrame()), fref, encoding
        )
        confusion_matrix = frames.get("confusion_matrix", pd.DataFrame())
        stats = _decode_stats(frames.get("stats", pd.DataFrame()))
        roc = _decode_roc(frames.get("roc", pd.DataFrame()))

        return PropensityTrainResult(
            model_id=model_id,
            features=features,
            confusion_matrix=confusion_matrix,
            stats=stats,
            roc=roc,
        )


def _decode_features(
    features: pd.DataFrame,
    fref: dict[str, dict[str, str]],
    encoding: bool,
) -> pd.DataFrame:
    if features.empty:
        return features
    features["Importance"] = features["Importance"].astype("float")
    features.sort_values(by=["Importance"], ascending=False, inplace=True)
    if encoding:
        for idx, _ in features.iterrows():
            features.loc[idx, "Feature"] = FieldNameEncoder.decode_value(
                features.loc[idx, "Feature"],
                fref,
            )
    return features


def _decode_stats(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return stats
    stats["Value"] = stats["Value"].astype("float")
    return stats


def _decode_roc(roc: pd.DataFrame) -> pd.DataFrame:
    if roc.empty:
        return roc
    if "TPR" in roc.keys():
        roc["TPR"] = roc["TPR"].astype(float)
    if "FPR" in roc.keys():
        roc["FPR"] = roc["FPR"].astype(float)
        roc = roc.sort_values(by=["FPR"], ascending=True)
    return roc
