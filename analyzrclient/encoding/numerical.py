"""Z-score normalization encoder for numerical columns.

Each column is standardized to zero mean and unit variance using statistics
captured during the initial encode call (zref).  The same statistics are used
for decoding, guaranteeing round-trip fidelity.  First-derivative decoding is
provided for converting regression coefficients between z-scored spaces.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class NumericalEncoder:
    """Encodes/decodes numerical values using z-score normalization (zref)."""

    @staticmethod
    def encode(
        series: pd.Series, numerical: bool = True
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Replace numerical values with z-scores, generating a new zref key map.

        :param series: Numerical series to normalize.
        :param numerical: Apply z-score normalization when ``True``; return series unchanged when ``False``.
        :return: Tuple of (z-scored series, zref dict containing ``mean`` and ``stdev``).
        :rtype: tuple[pd.Series, dict]
        """
        zref: dict[str, Any] = {"mean": series.mean(), "stdev": series.std()}
        series2 = NumericalEncoder.encode_with_keys(series, zref, numerical=numerical)
        return series2, zref

    @staticmethod
    def encode_with_keys(
        series: pd.Series,
        zref: dict[str, Any],
        numerical: bool = True,
    ) -> pd.Series:
        """Replace numerical values with z-scores using an existing zref key map.

        Returns the original series unchanged when ``numerical`` is ``False`` or
        when the stored mean is NaN.  Division by zero is guarded — if stdev is
        zero or NaN the mean-subtracted series is returned as-is.

        :param series: Numerical series to normalize.
        :param zref: Encoding keys dict containing ``mean`` and ``stdev`` values.
        :param numerical: Apply z-score normalization when ``True``.
        :return: Z-scored (or unchanged) series.
        :rtype: pd.Series
        """
        if not numerical:
            return series
        series2 = deepcopy(series)
        if not np.isnan(zref["mean"]):
            series2 -= zref["mean"]
        else:
            log.error("Series mean is nan, cannot encode series")
            return series
        if not np.isnan(zref["stdev"]) and zref["stdev"] != 0.0:
            series2 /= zref["stdev"]
        return series2

    @staticmethod
    def decode(series: pd.Series, zref: dict[str, Any]) -> pd.Series:
        """Denormalize a z-scored series back to its original scale.

        :param series: Z-scored series to denormalize.
        :param zref: Encoding keys dict containing ``mean`` and ``stdev`` values.
        :return: Denormalized series.
        :rtype: pd.Series
        """
        series2 = deepcopy(series)
        if not np.isnan(zref["stdev"]) and zref["stdev"] != 0.0:
            series2 *= zref["stdev"]
        if not np.isnan(zref["mean"]):
            series2 += zref["mean"]
        return series2

    @staticmethod
    def decode_single(val: float, zref: dict[str, Any]) -> float:
        """Denormalize a single z-scored scalar back to its original scale.

        :param val: Z-scored scalar value.
        :param zref: Encoding keys dict containing ``mean`` and ``stdev`` values.
        :return: Denormalized scalar value.
        :rtype: float
        """
        result = val
        if not np.isnan(zref["stdev"]) and zref["stdev"] != 0.0:
            result *= zref["stdev"]
        if not np.isnan(zref["mean"]):
            result += zref["mean"]
        return result

    @staticmethod
    def decode_first_derivative(
        val: float,
        zref_x: dict[str, Any],
        zref_y: dict[str, Any],
    ) -> float:
        """Convert a coefficient from z-scored space back to original-scale units.

        Applies the chain rule: ``coef_original = coef_z * (stdev_y / stdev_x)``.
        Returns ``val`` unchanged when either stdev is zero or NaN.

        :param val: Coefficient value in z-scored space.
        :param zref_x: Encoding keys for the predictor variable (must contain ``stdev``).
        :param zref_y: Encoding keys for the outcome variable (must contain ``stdev``).
        :return: Coefficient rescaled to original units.
        :rtype: float
        """
        new_val = val
        if (
            not np.isnan(zref_y["stdev"])
            and zref_y["stdev"] > 0.0
            and not np.isnan(zref_x["stdev"])
            and zref_x["stdev"] > 0.0
        ):
            new_val *= zref_y["stdev"] / zref_x["stdev"]
        return new_val
