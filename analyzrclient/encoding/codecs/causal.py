"""Codec for causal inference result decoding.

Decodes average treatment effect (ATX), raw group statistics, miscellaneous
metrics, and bin count frames.  Only mean-type estimators (ATC/ATT/ATE matched)
are homomorphic under z-score normalization; all other statistics are set to
``None`` during decode to prevent reporting non-invertible values.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...configs.causal import CausalTrainConfig
from ...results.causal import CausalTrainResult
from ..domain import DomainCodec
from ..numerical import NumericalEncoder


class CausalCodec(DomainCodec):
    """Decodes causal analysis results. Only averages are homomorphic."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        """Return buffer frame names required to decode causal train results.

        :param config: Causal training configuration (unused; included for interface parity).
        :return: List of frame names: ``atx``, ``raw``, ``misc``, ``bins``.
        :rtype: list[str]
        """
        return ["atx", "raw", "misc", "bins"]

    def decode_train_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> CausalTrainResult:
        """Decode causal buffer frames into a ``CausalTrainResult``.

        Only matched average treatment effect statistics (ATC/ATT/ATE) are
        denormalized; all other statistics in ``atx`` and ``raw`` are set to
        ``None`` when encoding is active.

        :param frames: Buffer frames: ``atx``, ``raw``, ``misc``, ``bins``.
        :param keys: Encoding keys; uses ``fref_exp`` for variable names and
            ``zref`` for outcome denormalization.
        :param config: Causal training configuration; provides ``outcome_var``.
        :param model_id: Unique identifier for the trained model.
        :param encoding: Apply denormalization and field-name decode when ``True``.
        :return: Decoded causal train result.
        :rtype: CausalTrainResult
        """
        fref = keys.get("fref_exp", {})
        zref = keys.get("zref", {})
        outcome_var = (
            config.outcome_var if isinstance(config, CausalTrainConfig) else None
        )

        atx = frames.get("atx", pd.DataFrame())
        if not atx.empty:
            atx["Value"] = atx["Value"].astype("float")
        if encoding:
            atx = _decode_atx(atx, fref, zref, outcome_var)

        raw = frames.get("raw", pd.DataFrame())
        if encoding:
            raw = _decode_raw(raw, fref, zref, outcome_var)
        else:
            for col in raw.columns:
                if col != "variable":
                    raw[col] = raw[col].astype("float")

        misc = frames.get("misc", pd.DataFrame())
        if not misc.empty:
            misc["Value"] = misc["Value"].astype("float")

        bins = frames.get("bins", pd.DataFrame())
        if not bins.empty:
            bins["count_treated"] = bins["count_treated"].astype("int")
            bins["count_untreated"] = bins["count_untreated"].astype("int")

        return CausalTrainResult(
            model_id=model_id, atx=atx, raw=raw, misc=misc, bins=bins
        )


def _decode_atx(
    atx: pd.DataFrame,
    fref: dict[str, dict[str, str]],
    zref: dict[str, Any],
    outcome_var: str | None,
) -> pd.DataFrame:
    """Denormalize ATX statistics; non-invertible stats (std, percentiles) are set to None."""
    atx2 = atx.copy()
    for idx, _ in atx2.iterrows():
        if outcome_var and outcome_var in zref:
            parameter = atx2.loc[idx, "Parameter"]
            if parameter in ("atc_matched", "att_matched", "ate_matched"):
                atx2.loc[idx, "Value"] = NumericalEncoder.decode_single(
                    float(atx2.loc[idx, "Value"]),
                    zref[outcome_var],
                )
            else:
                atx2.loc[idx, "Value"] = None
        else:
            atx2.loc[idx, "Value"] = float(atx2.loc[idx, "Value"])
    return atx2


def _decode_raw(
    raw: pd.DataFrame,
    fref: dict[str, dict[str, str]],
    zref: dict[str, Any],
    outcome_var: str | None,
) -> pd.DataFrame:
    """Denormalize raw group statistics; non-invertible columns (stdev, normalized diff) are set to None."""
    columns = [
        "control_mean",
        "treatment_mean",
        "control_stdev",
        "treatment_stdev",
        "difference_raw",
        "difference_normalized",
        "variable",
    ]
    raw2 = raw.copy()
    reverse = fref.get("reverse", {})
    raw2["variable"] = raw2["variable"].apply(
        lambda s, rev=reverse: rev[s] if s in rev else s,
    )
    for i in range(len(raw2)):
        idx = raw2.index[i]
        variable = raw2.at[idx, "variable"]
        if variable in zref:
            for col in columns:
                if col in ("control_mean", "treatment_mean"):
                    raw2.at[idx, col] = NumericalEncoder.decode_single(
                        float(raw2.at[idx, col]),
                        zref[variable],
                    )
                elif col == "difference_raw":
                    raw2.at[idx, col] = float(raw2.at[idx, "treatment_mean"]) - float(
                        raw2.at[idx, "control_mean"]
                    )
                elif col != "variable":
                    raw2.at[idx, col] = None
        else:
            for col in columns:
                if col != "variable":
                    val = raw2.at[idx, col]
                    if val is not None:
                        raw2.at[idx, col] = float(val)
    return raw2
