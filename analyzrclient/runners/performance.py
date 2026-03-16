from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from ..configs.performance import (
    PerformanceModelConfig,
    PerformanceReadConfig,
    PerformanceRunConfig,
)
from ..encoding.codecs.performance import PerformanceCodec
from ..exceptions import AnalyzrError
from ..results.performance import PerformanceRunResult, PerformanceTrainResult
from .base import BaseRunner

log = logging.getLogger(__name__)


class PerformanceRunner(BaseRunner):
    """Runs the performance analysis pipeline."""

    _uri: str
    _perf_codec: PerformanceCodec

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=PerformanceCodec())
        self._perf_codec = PerformanceCodec()
        self._uri = f'{self._base_url}/analytics/'

    def train(
        self, df: pd.DataFrame, config: PerformanceModelConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000, verbose: bool = False,
        timeout: int = 600, step: int = 2, poll: bool = True,
        compressed: bool = False, staging: bool = True,
        encoding: bool = True,
    ) -> PerformanceTrainResult:
        """Train a performance analysis model."""
        request_id = self._get_request_id()
        if verbose:
            log.info('Model ID: %s', request_id)

        fref: dict[str, Any] = {}
        xref: dict[str, Any] = {}

        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, categorical_vars=config.dimensional_vars,
                numerical_vars=config.primary_vars, bool_vars=[],
                record_id_var=config.idx_var, verbose=verbose, numerical=False,
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

        analysis: dict[str, Any] = {}

        if res['batches_saved'] == res['total_batches']:
            self._client.post(self._uri, {
                'command': 'analyze-train',
                'request_id': request_id,
                'client_id': client_id,
                'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
                'time_field': fref['forward'][config.time_var] if encoding else config.time_var,
                'outcome_var': fref['forward'][config.outcome_var] if encoding else config.outcome_var,
                'primary_fields': [fref['forward'][v] for v in config.primary_vars] if encoding else config.primary_vars,
                'dimensional_fields': [fref['forward'][v] for v in config.dimensional_vars] if encoding else config.dimensional_vars,
                'edges': self._perf_codec.encode_edges(config.edges, fref) if encoding else config.edges,
                'hierarchies': self._perf_codec.encode_hierarchies(config.hierarchies, fref) if encoding else config.hierarchies,
                'udf': self._perf_codec.encode_udf(config.udf, fref) if encoding else config.udf,
                'coef': self._perf_codec.encode_coefs(config.coef, fref) if encoding else config.coef,
                'staging': staging,
            })
            if poll:
                res2 = self._poller.poll(
                    payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
                    timeout=timeout, step=step, verbose=verbose,
                )
                if res2.get('response', {}).get('status') == 'Complete':
                    read_config = PerformanceReadConfig(
                        outcome_var=fref['forward'][config.outcome_var] if encoding else config.outcome_var,
                    )
                    raw_analysis = self._read_performance_analysis(
                        request_id=request_id, client_id=client_id, read_config=read_config,
                    )
                    if self._codec is None:
                        log.error('No codec configured for performance runner')
                        raise AnalyzrError('No codec configured for performance runner')
                    encode_keys = {'fref': fref if encoding else {}, 'xref': xref if encoding else {}}
                    result = self._codec.decode_train_results(
                        {}, encode_keys, config, request_id, encoding,
                        raw_analysis=raw_analysis,
                    )
                    analysis = result.analysis
                else:
                    log.warning('Training returned status: %s', res2.get('response', {}).get('status'))
        else:
            log.error('Buffer save failed: %s', res)
            raise AnalyzrError('Buffer save failed', detail=f'request_id={request_id}, response={res}')

        if poll:
            self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        return PerformanceTrainResult(model_id=request_id, analysis=analysis)

    def run(
        self, df: pd.DataFrame | None, model_id: str,
        client_id: str, run_config: PerformanceRunConfig,
        config: PerformanceModelConfig | None = None,
        buffer_batch_size: int = 1000, verbose: bool = False,
        timeout: int = 600, step: int = 2,
        compressed: bool = False, staging: bool = True,
        encoding: bool = True,
    ) -> PerformanceRunResult:
        """Run a pre-trained performance model, optionally with refreshed data."""
        if config is None:
            log.error('PerformanceModelConfig is required for run()')
            raise AnalyzrError('PerformanceModelConfig is required for run()', detail=f'model_id={model_id}')

        request_id = self._get_request_id()
        refresh_data = df is not None and not df.empty
        fref: dict[str, Any] = {}
        xref: dict[str, Any] = {}

        if refresh_data:
            df_non_null = cast(pd.DataFrame, df)
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    log.error('Keys not found for model_id=%s', model_id)
                    raise AnalyzrError('Keys not found', detail=f'model_id={model_id}')
                data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                    df_non_null, keys=keys, categorical_vars=config.dimensional_vars,
                    numerical_vars=config.primary_vars, bool_vars=[],
                    record_id_var=config.idx_var, verbose=verbose, numerical=False,
                )
            else:
                data = df_non_null

            res = self._buffer.save(
                data, client_id=client_id, request_id=request_id,
                verbose=verbose, batch_size=buffer_batch_size,
                compressed=compressed, staging=staging,
            )
            if res['batches_saved'] != res['total_batches']:
                log.error('Buffer save failed: %s', res)
                raise AnalyzrError('Buffer save failed', detail=f'request_id={request_id}, response={res}')
        else:
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    log.error('Keys not found for model_id=%s', model_id)
                    raise AnalyzrError('Keys not found', detail=f'model_id={model_id}')
                fref = keys['fref']
                xref = keys['xref']

        address = self._perf_codec.encode_address(
            list(run_config.address.values()), config.dimensional_vars, fref, xref,
        ) if encoding else tuple(run_config.address.values())

        encoded_outcome = fref['forward'][run_config.outcome_var] if encoding else run_config.outcome_var

        run_payload: dict[str, Any] = {
            'command': 'analyze-run',
            'request_id': request_id,
            'model_id': model_id,
            'client_id': client_id,
            'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
            'time_field': fref['forward'][config.time_var] if encoding else config.time_var,
            'outcome_var': encoded_outcome,
            'address': address,
            'main_stat': run_config.main_stat,
            'primary_fields': [fref['forward'][v] for v in config.primary_vars] if encoding else config.primary_vars,
            'dimensional_fields': [fref['forward'][v] for v in config.dimensional_vars] if encoding else config.dimensional_vars,
            'staging': staging,
            'refresh_data': refresh_data,
        }
        if run_config.period:
            run_payload['period'] = run_config.period
        if run_config.fiscal:
            run_payload.update(run_config.fiscal.to_payload())

        self._client.post(self._uri, run_payload)

        res2 = self._poller.poll(
            payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
            timeout=timeout, step=step, verbose=verbose,
        )

        analysis: dict[str, Any] = {}
        if res2.get('response', {}).get('status') == 'Complete':
            read_config = PerformanceReadConfig(
                outcome_var=encoded_outcome,
                main_stat=run_config.main_stat,
                address=address,
                period=run_config.period,
                fiscal=run_config.fiscal,
            )
            raw_analysis = self._read_performance_analysis(
                request_id=model_id, client_id=client_id, read_config=read_config,
            )
            if self._codec is None:
                log.error('No codec configured for performance runner')
                raise AnalyzrError('No codec configured for performance runner')
            encode_keys = {'fref': fref if encoding else {}, 'xref': xref if encoding else {}}
            result = self._codec.decode_train_results(
                {}, encode_keys, config, request_id, encoding,
                raw_analysis=raw_analysis,
            )
            analysis = result.analysis
        else:
            log.warning('Run returned status: %s', res2.get('response', {}).get('status'))

        if refresh_data:
            self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        return PerformanceRunResult(
            request_id=request_id, model_id=model_id, analysis=analysis,
        )

    def _read_performance_analysis(
        self, request_id: str, client_id: str | None,
        read_config: PerformanceReadConfig,
    ) -> dict[str, Any]:
        """Read performance analysis results via the API's cache resolution."""
        payload: dict[str, Any] = {
            'command': 'read-performance-analysis',
            'request_id': request_id,
            'client_id': client_id,
            'outcome_var': read_config.outcome_var,
            'address': list(read_config.address.values()) if isinstance(read_config.address, dict) else read_config.address,
            'period': read_config.period,
            'main_stat': read_config.main_stat,
        }
        if read_config.fiscal:
            payload.update(read_config.fiscal.to_payload())

        res = self._client.post(self._uri, payload)
        if res.get('status') == 200 and res.get('response') is not None:
            return res['response']
        log.error('Failed to read performance analysis: %s', res)
        raise AnalyzrError('Failed to read performance analysis', detail=f'request_id={request_id}, response={res}')

    def purge(
        self, model_id: str | None = None, client_id: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Purge data from a performance analysis model."""
        if verbose:
            log.info('Purging performance model data...')
        res = self._client.post(self._uri, {
            'command': 'analyze-purge',
            'model_id': model_id,
            'client_id': client_id,
        })
        if res['status'] != 200:
            log.error('Could not purge model: %s', res)
            raise AnalyzrError('Could not purge model', detail=f'model_id={model_id}, response={res}')
        return res['response']
