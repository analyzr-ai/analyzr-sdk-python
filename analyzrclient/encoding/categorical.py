"""UUID-based encoder for categorical columns.

Each unique categorical value is assigned a UUID string.  The bidirectional
forward/reverse map (xref) enables lossless decoding of model outputs back to
original category labels.  Values not seen during the initial encode are assigned
new UUIDs and logged as warnings.
"""

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
        """Replace categorical values with UUIDs, generating a new xref key map.

        :param series: Categorical series to encode.
        :return: Tuple of (UUID-encoded series, xref dict with forward and reverse maps).
        :rtype: tuple[pd.Series, dict]
        """
        values = series.value_counts().index
        xref: dict[str, dict[str, str]] = {"forward": {}, "reverse": {}}
        for val in values:
            if val is not None and val is not np.nan:
                key = str(uuid.uuid4())
                xref["forward"][str(val)] = key
                xref["reverse"][key] = str(val)
        series2, xref = CategoricalEncoder.encode_with_keys(series, xref)
        return series2, xref

    @staticmethod
    def encode_with_keys(
        series: pd.Series,
        xref: dict[str, dict[str, str]],
    ) -> tuple[pd.Series, dict[str, dict[str, str]]]:
        """Replace categorical values with UUIDs using an existing xref key map.

        Values absent from the training xref receive new UUIDs and are added
        to the map in-place; a warning is logged listing all such values.

        :param series: Categorical series to encode.
        :param xref: Existing encoding keys with forward and reverse maps; mutated in-place
            when new values are encountered.
        :return: Tuple of (UUID-encoded series, updated xref dict).
        :rtype: tuple[pd.Series, dict]
        """
        series2 = deepcopy(series)
        skipped_vals: list[str] = []
        for idx, val in series2.items():
            if val is not None and val is not np.nan:
                if str(val) not in xref["forward"]:
                    key = str(uuid.uuid4())
                    xref["forward"][str(val)] = key
                    xref["reverse"][key] = str(val)
                    if str(val) not in skipped_vals:
                        skipped_vals.append(str(val))
                series2[idx] = xref["forward"][str(val)]
            else:
                series2[idx] = None
        if len(skipped_vals) > 0:
            log.warning("Values not present in training encoding set: %s", skipped_vals)
        return series2, xref

    @staticmethod
    def decode(series: pd.Series, xref: dict[str, dict[str, str]]) -> pd.Series:
        """Replace UUID values with original categorical labels using the reverse map.

        :param series: UUID-encoded series to decode.
        :param xref: Encoding keys dict with a ``reverse`` map from UUID to original value.
        :return: Series with original categorical values restored.
        :rtype: pd.Series
        :raises KeyError: When a UUID in the series is absent from the reverse map.
        """
        series2 = deepcopy(series)
        for idx, val in series.items():
            series2[idx] = xref["reverse"][str(val)]
        return series2
