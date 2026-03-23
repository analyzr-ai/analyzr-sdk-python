"""Codec for cluster analysis result decoding.

Handles both in-memory and out-of-core clustering result layouts.  For standard
runs it merges decoded cluster assignments back to the original (unencoded)
DataFrame so callers receive a fully labeled result set.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...configs.cluster import ClusterTrainConfig
from ...results.cluster import ClusterTrainResult
from ..data_encoder import DataEncoder
from ..domain import DomainCodec


class ClusterCodec(DomainCodec):
    """Decodes cluster analysis results."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        """Return buffer frame names required to decode cluster train results.

        Returns ``stats_features`` instead of ``distances`` for out-of-core runs.

        :param config: Cluster training configuration; inspected for ``out_of_core`` flag.
        :return: List of buffer frame name strings.
        :rtype: list[str]
        """
        if isinstance(config, ClusterTrainConfig) and config.out_of_core:
            return ["res", "stats_mean", "stats_features"]
        return ["res", "stats_mean", "distances"]

    def decode_train_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> ClusterTrainResult:
        """Decode cluster buffer frames and merge assignments back to the original data.

        For out-of-core runs returns decoded cluster assignments and cluster-mean stats
        without merging.  For standard runs merges decoded cluster IDs back to the
        original unencoded DataFrame provided via ``original_df`` in ``kwargs``.

        :param frames: Buffer frames: ``res`` (assignments), ``stats_mean``, and either
            ``distances`` or ``stats_features``.
        :param keys: Encoding keys (xref, zref, rref, fref) for decoding ``res``.
        :param config: Cluster training configuration.
        :param model_id: Unique identifier for the trained model.
        :param encoding: Apply full DataEncoder decode on the result frame when ``True``.
        :return: Decoded cluster train result.
        :rtype: ClusterTrainResult
        """
        if not isinstance(config, ClusterTrainConfig):
            return ClusterTrainResult(model_id=model_id)

        res = frames.get("res", pd.DataFrame())
        if res.empty:
            return ClusterTrainResult(model_id=model_id)

        if encoding:
            decoded = DataEncoder.decode(
                res,
                categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars,
                record_id_var=config.idx_var,
                xref=keys.get("xref"),
                zref=keys.get("zref"),
                rref=keys.get("rref"),
                fref=keys.get("fref"),
            )
        else:
            decoded = res

        if config.out_of_core:
            stats_mean = frames.get("stats_mean")
            return ClusterTrainResult(
                model_id=model_id,
                data=decoded if decoded is not None and not decoded.empty else None,
                stats=stats_mean
                if stats_mean is not None and not stats_mean.empty
                else None,
            )

        original_df: pd.DataFrame | None = kwargs.get("original_df")
        if original_df is None or decoded is None:
            return ClusterTrainResult(model_id=model_id)

        decoded.reset_index(inplace=True)
        idx_var = config.idx_var or ""
        decoded[idx_var] = decoded[idx_var].astype(original_df[idx_var].dtype)
        merged = pd.merge(
            original_df, decoded, left_on=idx_var, right_on=idx_var, how="left"
        )

        stats = frames.get("stats_mean")
        distances = frames.get("distances")

        return ClusterTrainResult(
            model_id=model_id,
            data=merged,
            stats=stats if stats is not None and not stats.empty else None,
            distances=distances
            if distances is not None and not distances.empty
            else None,
        )
