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
    def encode(series: pd.Series, numerical: bool = True) -> tuple[pd.Series, dict[str, Any]]:
        """Replace numerical values with z-scores, generating new keys."""
        zref: dict[str, Any] = {'mean': series.mean(), 'stdev': series.std()}
        series2 = NumericalEncoder.encode_with_keys(series, zref, numerical=numerical)
        return series2, zref

    @staticmethod
    def encode_with_keys(
        series: pd.Series, zref: dict[str, Any], numerical: bool = True,
    ) -> pd.Series:
        """Replace numerical values with z-scores using existing keys."""
        if not numerical:
            return series
        series2 = deepcopy(series)
        if not np.isnan(zref['mean']):
            series2 -= zref['mean']
        else:
            log.error('Series mean is nan, cannot encode series')
            return series
        if not np.isnan(zref['stdev']) and zref['stdev'] != 0.0:
            series2 /= zref['stdev']
        return series2

    @staticmethod
    def decode(series: pd.Series, zref: dict[str, Any]) -> pd.Series:
        """Replace z-scores with denormalized numerical values."""
        series2 = deepcopy(series)
        if not np.isnan(zref['stdev']) and zref['stdev'] != 0.0:
            series2 *= zref['stdev']
        if not np.isnan(zref['mean']):
            series2 += zref['mean']
        return series2

    @staticmethod
    def decode_single(val: float, zref: dict[str, Any]) -> float:
        """Decode a single z-scored value back to original scale."""
        result = val
        if not np.isnan(zref['stdev']) and zref['stdev'] != 0.0:
            result *= zref['stdev']
        if not np.isnan(zref['mean']):
            result += zref['mean']
        return result

    @staticmethod
    def decode_first_derivative(
        val: float, zref_x: dict[str, Any], zref_y: dict[str, Any],
    ) -> float:
        """Decode first derivative (e.g. regression coefficients) between two z-scored variables."""
        new_val = val
        if (not np.isnan(zref_y['stdev']) and zref_y['stdev'] > 0.0
                and not np.isnan(zref_x['stdev']) and zref_x['stdev'] > 0.0):
            new_val *= (zref_y['stdev'] / zref_x['stdev'])
        return new_val
