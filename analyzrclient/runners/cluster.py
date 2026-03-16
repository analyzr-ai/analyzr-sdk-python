from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..configs.cluster import ClusterPredictConfig, ClusterTrainConfig
from ..encoding.codecs.cluster import ClusterCodec
from ..exceptions import AnalyzrError
from ..results.cluster import ClusterPredictResult, ClusterTrainResult
from .base import BaseRunner

log = logging.getLogger(__name__)


class ClusterRunner(BaseRunner):
    """Runs the clustering pipeline."""

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=ClusterCodec())
        self._uri = f'{self._base_url}/analytics/'

    def train(
        self, df: pd.DataFrame, config: ClusterTrainConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000, verbose: bool = False,
        timeout: int = 600, step: int = 2, poll: bool = True,
        compressed: bool = False, staging: bool = True,
        encoding: bool = True,
    ) -> ClusterTrainResult:
        """Train a clustering model."""
        request_id = self._get_request_id()
        if verbose:
            log.info('Request ID: %s', request_id)

        fref: dict[str, Any] = {}
        xref: dict[str, Any] = {}
        zref: dict[str, Any] = {}
        rref: dict[str, Any] = {}
        if encoding:
            keys = self._keys_load(model_id=request_id, verbose=verbose)
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, keys=keys, categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars, bool_vars=config.bool_vars,
                record_id_var=config.idx_var, verbose=verbose,
            )
            self._keys_save(
                model_id=request_id,
                keys={'xref': xref, 'zref': zref, 'rref': rref,
                      'fref': fref, 'fref_exp': fref_exp, 'bref': bref,
                      'idx_var': config.idx_var},
            )
        else:
            data = df

        res = self._buffer.save(
            data, client_id=client_id, request_id=request_id,
            verbose=verbose, batch_size=buffer_batch_size,
            compressed=compressed, staging=staging,
        )

        result = ClusterTrainResult(model_id=request_id)

        if res['batches_saved'] == res['total_batches']:
            self._client.post(self._uri, {
                'command': 'cluster-lazy' if config.algorithm == 'dask-pca-kmeans' else 'cluster-train',
                'request_id': request_id,
                'client_id': client_id,
                'algorithm': config.algorithm,
                'n_components': config.n_components,
                'batch_size': config.cluster_batch_size,
                'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
                'categorical_fields': [fref['forward'][v] for v in config.categorical_vars] if encoding else config.categorical_vars,
                'staging': staging,
                'out_of_core': config.out_of_core,
            })

            if poll:
                res2 = self._poller.poll(
                    payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
                    timeout=timeout, step=step, verbose=verbose,
                )
                if res2.get('response', {}).get('status') == 'Complete':
                    if self._codec is None:
                        log.error('No codec configured for cluster runner')
                        raise AnalyzrError('No codec configured for cluster runner')
                    frame_names = self._codec.get_train_frame_names(config)
                    frames = self._read_frames(
                        frame_names, request_id, client_id,
                        verbose=verbose, staging=True,
                    )
                    encode_keys = {'xref': xref, 'zref': zref, 'rref': rref, 'fref': fref}
                    result = self._codec.decode_train_results(
                        frames, encode_keys, config, request_id, encoding,
                        original_df=df,
                    )
                else:
                    log.warning('Training returned status: %s', res2.get('response', {}).get('status'))
        else:
            log.error('Buffer save failed: %s', res)
            raise AnalyzrError('Buffer save failed', detail=f'request_id={request_id}, response={res}')

        if poll:
            self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose, out_of_core=config.out_of_core)

        return result

    def predict(
        self, df: pd.DataFrame, config: ClusterPredictConfig,
        model_id: str | None = None, client_id: str | None = None,
        buffer_batch_size: int = 1000, timeout: int = 600,
        verbose: bool = False, compressed: bool = False,
        staging: bool = True, encoding: bool = True,
    ) -> ClusterPredictResult:
        """Assign cluster IDs using a pre-trained model."""
        request_id = self._get_request_id()
        fref: dict[str, Any] = {}

        if encoding:
            keys = self._keys_load(model_id=model_id or '', verbose=verbose)
            if keys is None:
                log.error('Keys not found for model_id=%s', model_id)
                raise AnalyzrError('Keys not found', detail=f'model_id={model_id}')
            data, xref, zref, rref, fref, _fref_exp, bref = self._encode(
                df, keys=keys, categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars, bool_vars=config.bool_vars,
                record_id_var=config.idx_var, verbose=verbose,
            )
        else:
            data = df
            xref, zref, rref, bref = {}, {}, {}, {}

        res = self._buffer.save(
            data, client_id=client_id, request_id=request_id,
            verbose=verbose, batch_size=buffer_batch_size,
            compressed=compressed, staging=staging,
        )

        data2: pd.DataFrame | None = None
        if res['batches_saved'] == res['total_batches']:
            self._client.post(self._uri, {
                'command': 'cluster-predict',
                'model_id': model_id,
                'request_id': request_id,
                'client_id': client_id,
                'batch_size': config.cluster_batch_size,
                'idx_field': fref['forward'][config.idx_var] if encoding and config.idx_var else config.idx_var,
                'categorical_fields': [fref['forward'][v] for v in config.categorical_vars] if encoding else config.categorical_vars,
                'staging': staging,
            })
            self._poller.poll(
                payload={'request_id': request_id, 'client_id': client_id, 'command': 'task-status'},
                timeout=timeout, step=2, verbose=verbose,
            )
            data2 = self._buffer.read(
                request_id=request_id, client_id=client_id,
                dataframe_name='res', verbose=verbose, staging=True,
            )
        else:
            log.error('Buffer save failed: %s', res)
            raise AnalyzrError('Buffer save failed', detail=f'request_id={request_id}, response={res}')

        self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        if encoding and data2 is not None:
            data2 = self._decode(
                data2, categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars, bool_vars=config.bool_vars,
                record_id_var=config.idx_var, xref=xref, zref=zref,
                rref=rref, fref=fref, bref=bref, verbose=verbose,
            )

        return ClusterPredictResult(model_id=model_id or '', data=data2)
