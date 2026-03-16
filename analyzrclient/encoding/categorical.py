from __future__ import annotations

import logging
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class CategoricalEncoder:
    """Encodes/decodes categorical values using UUID mapping (xref)."""

    @staticmethod
    def encode(series: pd.Series) -> tuple[pd.Series, dict[str, dict[str, str]]]:
        """Replace categorical values with UUIDs, generating new keys."""
        values = series.value_counts().index
        xref: dict[str, dict[str, str]] = {'forward': {}, 'reverse': {}}
        for val in values:
            if val is not None and val is not np.nan:
                key = str(uuid.uuid4())
                xref['forward'][str(val)] = key
                xref['reverse'][key] = str(val)
        series2, xref = CategoricalEncoder.encode_with_keys(series, xref)
        return series2, xref

    @staticmethod
    def encode_with_keys(
        series: pd.Series, xref: dict[str, dict[str, str]],
    ) -> tuple[pd.Series, dict[str, dict[str, str]]]:
        """Replace categorical values with UUIDs using existing keys."""
        series2 = deepcopy(series)
        skipped_vals: list[str] = []
        for idx, val in series2.items():
            if val is not None and val is not np.nan:
                if str(val) not in xref['forward']:
                    key = str(uuid.uuid4())
                    xref['forward'][str(val)] = key
                    xref['reverse'][key] = str(val)
                    if str(val) not in skipped_vals:
                        skipped_vals.append(str(val))
                series2[idx] = xref['forward'][str(val)]
            else:
                series2[idx] = None
        if len(skipped_vals) > 0:
            log.warning('Values not present in training encoding set: %s', skipped_vals)
        return series2, xref

    @staticmethod
    def decode(series: pd.Series, xref: dict[str, dict[str, str]]) -> pd.Series:
        """Replace UUIDs with original categorical values."""
        series2 = deepcopy(series)
        for idx, val in series.items():
            series2[idx] = xref['reverse'][str(val)]
        return series2
