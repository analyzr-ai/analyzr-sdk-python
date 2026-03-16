from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..configs.mmm import MMMOptimizeConfig, MMMTrainConfig
from ..encoding.codecs.mmm import MMMCodec
from ..exceptions import AnalyzrError
from ..results.mmm import MMMOptimizeResult, MMMTrainResult
from .base import BaseRunner

log = logging.getLogger(__name__)


class MMMRunner(BaseRunner):
    """Runs the marketing mix modeling pipeline."""

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=MMMCodec())
        self._uri = f'{self._base_url}/analytics/'

    def train(
        self, df: pd.DataFrame, config: MMMTrainConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000, verbose: bool = False,
        timeout: int = 600, step: int = 2, poll: bool = True,
        compressed: bool = False, staging: bool = True,
        encoding: bool = True,
    ) -> MMMTrainResult:
        """Train a marketing mix model."""
        request_id = self._get_request_id()
        if verbose:
            log.info('Model ID: %s', request_id)

        fref: dict[str, Any] = {}
        fref_exp: dict[str, Any] = {}
        zref: dict[str, Any] = {}

        if encoding:
            numerical_vars: list[str] = list(config.media_vars)
            if config.other_vars:
                numerical_vars = [*numerical_vars, *config.other_vars]
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, numerical_vars=numerical_vars, skip_vars=numerical_vars,
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

        result = MMMTrainResult(model_id=request_id)

        if res['batches_saved'] == res['total_batches']:
            self._client.post(self._uri, {
                'command': 'mmm-train',
                'request_id': request_id,
                'client_id': client_id,
                'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
                'time_field': fref['forward'][config.time_var] if encoding and config.time_var else config.time_var,
                'outcome_var': fref['forward'][config.outcome_var] if encoding and config.outcome_var else config.outcome_var,
                'media_fields': [fref['forward'][v] for v in config.media_vars] if encoding else config.media_vars,
                'other_fields': [fref['forward'][v] for v in config.other_vars] if encoding else config.other_vars,
                'staging': staging,
                'algorithm': config.algorithm,
            })
            if poll:
                res2 = self._poller.poll(
                    payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
                    timeout=timeout, step=step, verbose=verbose,
                )
                if res2.get('response', {}).get('status') == 'Complete':
                    if self._codec is None:
                        log.error('No codec configured for mmm runner')
                        raise AnalyzrError('No codec configured for mmm runner')
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

    def optimize(
        self, model_id: str, config: MMMOptimizeConfig,
        client_id: str | None = None,
        timeout: int = 600, step: int = 2, verbose: bool = False,
        encoding: bool = True,
    ) -> MMMOptimizeResult:
        """Optimize marketing mix model budget allocation."""
        request_id = self._get_request_id()
        fref: dict[str, Any] = {}

        if encoding:
            keys = self._keys_load(model_id=model_id, verbose=verbose)
            if keys is None:
                log.error('Keys not found for model_id=%s', model_id)
                raise AnalyzrError('Keys not found', detail=f'model_id={model_id}')
            fref = keys['fref']

        self._client.post(self._uri, {
            'command': 'mmm-optimize',
            'model_id': model_id,
            'request_id': request_id,
            'client_id': client_id,
            'budget': config.budget,
        })
        self._poller.poll(
            payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
            timeout=timeout, step=step, verbose=verbose,
        )
        data2 = self._buffer.read(
            request_id=request_id, client_id=client_id,
            dataframe_name='res', verbose=verbose, staging=True,
        )

        self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        data2 = MMMCodec.decode_optimize_results(data2, fref, encoding)

        return MMMOptimizeResult(model_id=model_id, data=data2)
