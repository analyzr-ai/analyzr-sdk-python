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
        return ['atx', 'raw', 'misc', 'bins']

    def decode_train_results(
        self, frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> CausalTrainResult:
        fref = keys.get('fref_exp', {})
        zref = keys.get('zref', {})
        outcome_var = config.outcome_var if isinstance(config, CausalTrainConfig) else None

        atx = frames.get('atx', pd.DataFrame())
        if not atx.empty:
            atx['Value'] = atx['Value'].astype('float')
        if encoding:
            atx = _decode_atx(atx, fref, zref, outcome_var)

        raw = frames.get('raw', pd.DataFrame())
        if encoding:
            raw = _decode_raw(raw, fref, zref, outcome_var)
        else:
            for col in raw.columns:
                if col != 'variable':
                    raw[col] = raw[col].astype('float')

        misc = frames.get('misc', pd.DataFrame())
        if not misc.empty:
            misc['Value'] = misc['Value'].astype('float')

        bins = frames.get('bins', pd.DataFrame())
        if not bins.empty:
            bins['count_treated'] = bins['count_treated'].astype('int')
            bins['count_untreated'] = bins['count_untreated'].astype('int')

        return CausalTrainResult(model_id=model_id, atx=atx, raw=raw, misc=misc, bins=bins)


def _decode_atx(
    atx: pd.DataFrame, fref: dict[str, dict[str, str]],
    zref: dict[str, Any], outcome_var: str | None,
) -> pd.DataFrame:
    """Only averages are homomorphic — other stats set to None."""
    atx2 = atx.copy()
    for idx, _ in atx2.iterrows():
        if outcome_var and outcome_var in zref:
            parameter = atx2.loc[idx, 'Parameter']
            if parameter in ('atc_matched', 'att_matched', 'ate_matched'):
                atx2.loc[idx, 'Value'] = NumericalEncoder.decode_single(
                    float(atx2.loc[idx, 'Value']), zref[outcome_var],
                )
            else:
                atx2.loc[idx, 'Value'] = None
        else:
            atx2.loc[idx, 'Value'] = float(atx2.loc[idx, 'Value'])
    return atx2


def _decode_raw(
    raw: pd.DataFrame, fref: dict[str, dict[str, str]],
    zref: dict[str, Any], outcome_var: str | None,
) -> pd.DataFrame:
    """Only averages are homomorphic — other stats set to None."""
    columns = ['control_mean', 'treatment_mean', 'control_stdev',
                'treatment_stdev', 'difference_raw', 'difference_normalized', 'variable']
    raw2 = raw.copy()
    reverse = fref.get('reverse', {})
    raw2['variable'] = raw2['variable'].apply(
        lambda s, rev=reverse: rev[s] if s in rev else s,
    )
    for i in range(len(raw2)):
        idx = raw2.index[i]
        variable = raw2.at[idx, 'variable']
        if variable in zref:
            for col in columns:
                if col in ('control_mean', 'treatment_mean'):
                    raw2.at[idx, col] = NumericalEncoder.decode_single(
                        float(raw2.at[idx, col]), zref[variable],
                    )
                elif col == 'difference_raw':
                    raw2.at[idx, col] = float(raw2.at[idx, 'treatment_mean']) - float(raw2.at[idx, 'control_mean'])
                elif col != 'variable':
                    raw2.at[idx, col] = None
        else:
            for col in columns:
                if col != 'variable':
                    val = raw2.at[idx, col]
                    if val is not None:
                        raw2.at[idx, col] = float(val)
    return raw2
