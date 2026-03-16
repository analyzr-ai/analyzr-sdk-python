from __future__ import annotations

from typing import Any

import pandas as pd

from ...results.mmm import MMMTrainResult
from ..domain import DomainCodec
from ..field_name import FieldNameEncoder


class MMMCodec(DomainCodec):
    """Decodes marketing mix model results."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        return [
            'train_stats', 'train_data', 'test_stats',
            'lag_stats', 'lag_hist', 'media_spend',
            'contrib_stats', 'contrib_data',
            'resp_curve_media', 'resp_curve_pred',
        ]

    def decode_train_results(
        self, frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> MMMTrainResult:
        fref = keys.get('fref_exp', {})

        train_stats = frames.get('train_stats', pd.DataFrame())
        if not train_stats.empty:
            train_stats['Value'] = train_stats['Value'].astype('float')

        train_data = frames.get('train_data', pd.DataFrame())
        if encoding and not train_data.empty:
            train_data.columns = FieldNameEncoder.decode_columns(train_data.columns, fref)

        test_stats = frames.get('test_stats', pd.DataFrame())
        if not test_stats.empty:
            test_stats['Value'] = test_stats['Value'].astype('float')

        lag_stats = frames.get('lag_stats', pd.DataFrame())
        for col in lag_stats.columns:
            if col != 'index':
                lag_stats[col] = lag_stats[col].astype('float')
        if encoding and not lag_stats.empty:
            lag_stats.columns = FieldNameEncoder.decode_columns(lag_stats.columns, fref)

        contrib_stats = frames.get('contrib_stats', pd.DataFrame())
        for col in contrib_stats.columns:
            if col not in ('stat', 'metric'):
                contrib_stats[col] = contrib_stats[col].astype('float')
        if encoding and not contrib_stats.empty:
            contrib_stats.columns = FieldNameEncoder.decode_columns(contrib_stats.columns, fref)

        lag_hist = frames.get('lag_hist', pd.DataFrame())
        if encoding and not lag_hist.empty:
            lag_hist.columns = FieldNameEncoder.decode_columns(lag_hist.columns, fref)
        media_spend = frames.get('media_spend', pd.DataFrame())
        if encoding and not media_spend.empty:
            media_spend.columns = FieldNameEncoder.decode_columns(media_spend.columns, fref)

        contrib_data = frames.get('contrib_data', pd.DataFrame())
        if encoding and not contrib_data.empty:
            contrib_data.columns = FieldNameEncoder.decode_columns(contrib_data.columns, fref)

        resp_curve_media = frames.get('resp_curve_media', pd.DataFrame())
        if encoding and not resp_curve_media.empty:
            resp_curve_media.columns = FieldNameEncoder.decode_columns(resp_curve_media.columns, fref)

        resp_curve_pred = frames.get('resp_curve_pred', pd.DataFrame())
        if encoding and not resp_curve_pred.empty:
            resp_curve_pred.columns = FieldNameEncoder.decode_columns(resp_curve_pred.columns, fref)

        return MMMTrainResult(
            model_id=model_id, train_data=train_data,
            train_stats=train_stats, test_stats=test_stats,
            lag_stats=lag_stats, lag_hist=lag_hist,
            media_spend=media_spend,
            contrib_stats=contrib_stats, contrib_data=contrib_data,
            resp_curve_media=resp_curve_media, resp_curve_pred=resp_curve_pred,
        )

    @staticmethod
    def decode_optimize_results(
        data: pd.DataFrame, fref: dict[str, Any], encoding: bool,
    ) -> pd.DataFrame:
        """Decode optimize results — field name decode on media_ parameters."""
        if encoding:
            for i, val in data['Parameter'].items():
                if str(val)[:6] == 'media_':
                    data.loc[i, 'Parameter'] = f'media_{FieldNameEncoder.decode_value(str(val)[6:], fref)}'
        return data
