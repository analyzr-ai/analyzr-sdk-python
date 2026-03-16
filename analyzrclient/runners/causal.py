from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..configs.causal import CausalTrainConfig
from ..encoding.codecs.causal import CausalCodec
from ..exceptions import AnalyzrError
from ..results.causal import CausalTrainResult
from .base import BaseRunner

log = logging.getLogger(__name__)


class CausalRunner(BaseRunner):
    """Runs the causal analysis pipeline."""

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=CausalCodec())
        self._uri = f'{self._base_url}/analytics/'

    def train(
        self, df: pd.DataFrame, config: CausalTrainConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000, verbose: bool = False,
        timeout: int = 600, step: int = 2, poll: bool = True,
        compressed: bool = False, staging: bool = True,
        encoding: bool = True,
    ) -> CausalTrainResult:
        """Train a causal analysis model."""
        request_id = self._get_request_id()
        if verbose:
            log.info('Model ID: %s', request_id)

        fref: dict[str, Any] = {}
        fref_exp: dict[str, Any] = {}
        zref: dict[str, Any] = {}

        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars, bool_vars=config.bool_vars,
                record_id_var=config.idx_var, verbose=verbose,
            )
            self._keys_save(
                model_id=request_id,
                keys={'xref': xref, 'zref': zref, 'rref': rref,
                      'fref': fref, 'fref_exp': fref_exp, 'bref': bref},
            )
        else:
            data = df

        res = self._buffer.save(
            data, client_id=client_id, request_id=request_id,
            verbose=verbose, batch_size=buffer_batch_size,
            compressed=compressed, staging=staging,
        )

        result = CausalTrainResult(model_id=request_id)

        if res['batches_saved'] == res['total_batches']:
            self._client.post(self._uri, {
                'command': 'causal-train',
                'request_id': request_id,
                'client_id': client_id,
                'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
                'outcome_var': fref['forward'][config.outcome_var] if encoding and config.outcome_var else config.outcome_var,
                'treatment_var': fref['forward'][config.treatment_var] if encoding and config.treatment_var else config.treatment_var,
                'categorical_fields': [fref['forward'][v] for v in config.categorical_vars] if encoding else config.categorical_vars,
                'staging': staging,
                'algorithm': config.algorithm,
                'standard_error': config.standard_error,
            })
            if poll:
                res2 = self._poller.poll(
                    payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
                    timeout=timeout, step=step, verbose=verbose,
                )
                if res2.get('response', {}).get('status') == 'Complete':
                    if self._codec is None:
                        log.error('No codec configured for causal runner')
                        raise AnalyzrError('No codec configured for causal runner')
                    frame_names = self._codec.get_train_frame_names(config)
                    frames = self._read_frames(
                        frame_names, request_id, client_id,
                        verbose=verbose, staging=True,
                    )
                    encode_keys = {'fref_exp': fref_exp if encoding else {}, 'zref': zref if encoding else {}}
                    result = self._codec.decode_train_results(
                        frames, encode_keys, config, request_id, encoding,
                    )
                else:
                    log.warning('Training returned status: %s', res2.get('response', {}).get('status'))
        else:
            log.error('Buffer save failed: %s', res)
            raise AnalyzrError('Buffer save failed', detail=f'request_id={request_id}, response={res}')

        if poll:
            self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        return result
