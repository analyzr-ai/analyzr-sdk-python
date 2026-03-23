"""Result dataclasses for causal inference workflows.

Provides ``CausalTrainResult`` which encapsulates the structured outputs
returned by the analyzr analytics API after causal analysis model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CausalTrainResult:
    """Result from causal analysis model training.

    Holds the model identifier and the DataFrames produced by the analytics
    engine after fitting a causal inference (treatment-effect estimation)
    model.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent requests.
    :param atx: DataFrame of average treatment effect (ATX) estimates,
        including point estimates and optional confidence intervals, or
        ``None`` if not returned by the API.
    :param raw: DataFrame of raw matched or weighted records used during
        treatment-effect estimation, or ``None`` if not returned.
    :param misc: DataFrame of miscellaneous diagnostics (e.g. covariate
        balance statistics, propensity score distributions), or ``None``
        if not returned.
    :param bins: DataFrame of treatment-effect estimates stratified by
        propensity score bins, or ``None`` if not returned.
    """

    model_id: str
    atx: pd.DataFrame | None = None
    raw: pd.DataFrame | None = None
    misc: pd.DataFrame | None = None
    bins: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id``, ``atx``, ``raw``, ``misc``,
            and ``bins``.
        :rtype: dict[str, Any]
        """
        return {
            "model_id": self.model_id,
            "atx": self.atx,
            "raw": self.raw,
            "misc": self.misc,
            "bins": self.bins,
        }
