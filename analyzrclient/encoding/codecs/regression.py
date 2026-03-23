"""Codec for regression analysis result decoding.

Decodes feature importance, model statistics, linear coefficients, and
carry/saturation parameter frames.  Coefficient values are additionally
rescaled from z-scored space back to original-scale units using the chain rule
implemented in ``NumericalEncoder.decode_first_derivative``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...configs.regression import LINEAR_ALGORITHMS, RegressionTrainConfig
from ...results.regression import RegressionTrainResult
from ..domain import DomainCodec
from ..field_name import FieldNameEncoder
from ..numerical import NumericalEncoder


class RegressionCodec(DomainCodec):
    """Decodes regression analysis results."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        """Return buffer frame names required to decode regression train results.

        Appends ``'coefs'`` for linear algorithms and ``'carrysats'`` when saturation
        or lagging variables are configured.

        :param config: Regression training configuration.
        :return: List of buffer frame name strings.
        :rtype: list[str]
        """
        if not isinstance(config, RegressionTrainConfig):
            return ["features", "stats"]
        names = ["features", "stats"]
        if config.algorithm in LINEAR_ALGORITHMS:
            names.append("coefs")
            if config.saturation_vars or config.lagging_vars:
                names.append("carrysats")
        return names

    def decode_train_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> RegressionTrainResult:
        """Decode regression buffer frames into a ``RegressionTrainResult``.

        :param frames: Buffer frames: ``features``, ``stats``, optionally ``coefs``
            and ``carrysats``.
        :param keys: Encoding keys; uses ``fref_exp`` for feature/coef names and
            ``zref`` for coefficient rescaling.
        :param config: Regression training configuration.
        :param model_id: Unique identifier for the trained model.
        :param encoding: Decode field names and rescale coefficients when ``True``.
        :return: Decoded regression train result.
        :rtype: RegressionTrainResult
        """
        if not isinstance(config, RegressionTrainConfig):
            return RegressionTrainResult(model_id=model_id)

        fref = keys.get("fref_exp", {})
        zref = keys.get("zref", {})

        features = _decode_features(
            frames.get("features", pd.DataFrame()), fref, encoding
        )
        stats = _decode_stats(frames.get("stats", pd.DataFrame()))
        coefs = _decode_coefs(
            frames.get("coefs"), fref, zref, config.outcome_var, encoding
        )
        carrysats = _decode_carrysats(frames.get("carrysats"), fref, encoding)

        return RegressionTrainResult(
            model_id=model_id,
            features=features,
            stats=stats,
            coefs=coefs,
            laggingsats=carrysats,
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


def _decode_coefs(
    coefs: pd.DataFrame | None,
    fref: dict[str, dict[str, str]],
    zref: dict[str, Any],
    outcome_var: str | None,
    encoding: bool,
) -> pd.DataFrame | None:
    if coefs is None or coefs.empty:
        return coefs
    coefs["Value"] = coefs["Value"].astype("float")
    if encoding:
        for idx, _ in coefs.iterrows():
            clear_var = FieldNameEncoder.decode_value(coefs.loc[idx, "Parameter"], fref)
            coefs.loc[idx, "Parameter"] = clear_var
            if clear_var in zref and outcome_var and outcome_var in zref:
                coefs.loc[idx, "Value"] = NumericalEncoder.decode_first_derivative(
                    coefs.loc[idx, "Value"],
                    zref[clear_var],
                    zref[outcome_var],
                )
    return coefs


def _decode_carrysats(
    carrysats: pd.DataFrame | None,
    fref: dict[str, dict[str, str]],
    encoding: bool,
) -> pd.DataFrame | None:
    if carrysats is None or carrysats.empty:
        return carrysats
    carrysats["Value"] = carrysats["Value"].astype("float")
    if encoding:
        for idx, _ in carrysats.iterrows():
            carrysats.loc[idx, "Variable"] = FieldNameEncoder.decode_value(
                carrysats.loc[idx, "Variable"],
                fref,
            )
    return carrysats
