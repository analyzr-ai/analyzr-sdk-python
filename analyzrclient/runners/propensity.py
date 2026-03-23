"""Runner for the propensity scoring pipeline (train and predict with API-level batching)."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import pandas as pd

from ..configs.propensity import PropensityPredictConfig, PropensityTrainConfig
from ..encoding.codecs.propensity import PropensityCodec
from ..exceptions import AnalyzrError
from ..results.propensity import PropensityPredictResult, PropensityTrainResult
from .base import BaseRunner

log = logging.getLogger(__name__)


class PropensityRunner(BaseRunner):
    """Runs the propensity scoring pipeline (train and predict).

    Prediction supports API-level batching via ``config.api_batch_size`` to
    handle large datasets that would otherwise exceed single-request limits.

    :param client: Authenticated SAML SSO client for API communication.
    :param base_url: Root URL of the Analyzr API tenant.
    """

    _uri: str

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=PropensityCodec())
        self._uri = f"{self._base_url}/analytics/"

    def train(
        self,
        df: pd.DataFrame,
        config: PropensityTrainConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000,
        verbose: bool = False,
        timeout: int = 600,
        step: int = 2,
        poll: bool = True,
        compressed: bool = False,
        staging: bool = True,
        encoding: bool = True,
    ) -> PropensityTrainResult:
        """Train a propensity model and optionally poll for evaluation results.

        :param df: Input training DataFrame.
        :param config: Training configuration (algorithm, variable lists, SMOTE flag,
                       cross-validation splits, scoring metric).
        :param client_id: Client/tenant identifier.
        :param buffer_batch_size: Number of rows per buffer upload batch.
        :param verbose: Enable verbose logging.
        :param timeout: Maximum seconds to wait for the job to complete.
        :param step: Polling interval in seconds.
        :param poll: Whether to block and poll until the job finishes.
        :param compressed: Whether to compress the buffer payload.
        :param staging: Whether to use the staging buffer area.
        :param encoding: Whether to encode field names and values before upload.
        :return: Training result containing the model ID and evaluation output.
        :rtype: PropensityTrainResult
        :raises AnalyzrError: If the buffer save fails or the codec is not configured.
        """
        request_id = self._get_request_id()
        if verbose:
            log.info("Model ID: %s", request_id)

        fref: dict[str, Any] = {}
        fref_exp: dict[str, Any] = {}

        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df,
                categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars,
                bool_vars=config.bool_vars,
                record_id_var=config.idx_var,
                verbose=verbose,
            )
            self._keys_save(
                model_id=request_id,
                keys={
                    "xref": xref,
                    "zref": zref,
                    "rref": rref,
                    "fref": fref,
                    "fref_exp": fref_exp,
                    "bref": bref,
                },
            )
        else:
            data = deepcopy(df)

        res = self._buffer.save(
            data,
            client_id=client_id,
            request_id=request_id,
            verbose=verbose,
            batch_size=buffer_batch_size,
            compressed=compressed,
            staging=staging,
        )

        result = PropensityTrainResult(model_id=request_id)

        if res["batches_saved"] == res["total_batches"]:
            self._client.post(
                self._uri,
                {
                    "command": "propensity-train",
                    "request_id": request_id,
                    "client_id": client_id,
                    "algorithm": config.algorithm,
                    "train_size": config.train_size,
                    "smote": config.smote,
                    "idx_field": fref["forward"][config.idx_var]
                    if encoding and config.idx_var
                    else config.idx_var,
                    "outcome_var": fref["forward"][config.outcome_var]
                    if encoding and config.outcome_var
                    else config.outcome_var,
                    "categorical_fields": [
                        fref["forward"][v] for v in config.categorical_vars
                    ]
                    if encoding
                    else config.categorical_vars,
                    "staging": staging,
                    "param_grid": config.param_grid,
                    "scoring": config.scoring,
                    "n_splits": config.n_splits,
                    "out_of_core": config.out_of_core,
                    "unique_categories": config.unique_categories,
                },
            )
            if poll:
                res2 = self._poller.poll(
                    payload={
                        "request_id": request_id,
                        "client_id": client_id,
                        "command": "task-status",
                    },
                    timeout=timeout,
                    step=step,
                    verbose=verbose,
                )
                if res2.get("response", {}).get("status") == "Complete":
                    if self._codec is None:
                        log.error("No codec configured for propensity runner")
                        raise AnalyzrError("No codec configured for propensity runner")
                    frame_names = self._codec.get_train_frame_names(config)
                    frames = self._read_frames(
                        frame_names,
                        request_id,
                        client_id,
                        verbose=verbose,
                        staging=True,
                    )
                    encode_keys = {"fref_exp": fref_exp if encoding else {}}
                    result = self._codec.decode_train_results(
                        frames,
                        encode_keys,
                        config,
                        request_id,
                        encoding,
                    )
                else:
                    log.warning(
                        "Training returned status: %s",
                        res2.get("response", {}).get("status"),
                    )
        else:
            log.error("Buffer save failed: %s", res)
            raise AnalyzrError(
                "Buffer save failed", detail=f"request_id={request_id}, response={res}"
            )

        if poll:
            self._buffer.clear(
                request_id=request_id, client_id=client_id, verbose=verbose
            )

        return result

    def predict(
        self,
        df: pd.DataFrame,
        config: PropensityPredictConfig,
        model_id: str | None = None,
        client_id: str | None = None,
        buffer_batch_size: int = 1000,
        verbose: bool = False,
        timeout: int = 600,
        step: int = 2,
        compressed: bool = False,
        staging: bool = True,
        encoding: bool = True,
    ) -> PropensityPredictResult:
        """Score records using a pre-trained propensity model with automatic API-level batching.

        Splits ``df`` into chunks of ``config.api_batch_size`` rows, sends each chunk
        as a separate API request, and concatenates the results before returning.

        :param df: Input DataFrame containing records to score.
        :param config: Prediction configuration including ``api_batch_size`` and variable lists.
        :param model_id: ID of the previously trained model.
        :param client_id: Client/tenant identifier.
        :param buffer_batch_size: Number of rows per buffer upload batch within each API batch.
        :param verbose: Enable verbose logging.
        :param timeout: Maximum seconds to wait per batch job.
        :param step: Polling interval in seconds.
        :param compressed: Whether to compress the buffer payload.
        :param staging: Whether to use the staging buffer area.
        :param encoding: Whether to encode/decode field names and values.
        :return: Prediction result containing the model ID and concatenated scored DataFrame.
        :rtype: PropensityPredictResult
        """
        result = PropensityPredictResult(model_id=model_id or "", data=pd.DataFrame())
        api_batch_size = config.api_batch_size
        if api_batch_size < buffer_batch_size:
            api_batch_size = buffer_batch_size

        batched_df: list[pd.DataFrame] = [
            df.iloc[i : i + api_batch_size] for i in range(0, len(df), api_batch_size)
        ]
        for idx, batch in enumerate(batched_df, 1):
            if verbose:
                log.info("Processing API request %s of %s", idx, len(batched_df))
            batch_result = self._predict_batch(
                batch,
                config=config,
                model_id=model_id,
                client_id=client_id,
                buffer_batch_size=buffer_batch_size,
                verbose=verbose,
                timeout=timeout,
                step=step,
                compressed=compressed,
                staging=staging,
                encoding=encoding,
            )
            if batch_result.data is not None and result.data is not None:
                result.data = pd.concat([result.data, batch_result.data])
        return result

    def _predict_batch(
        self,
        df: pd.DataFrame,
        config: PropensityPredictConfig,
        model_id: str | None,
        client_id: str | None,
        buffer_batch_size: int,
        verbose: bool,
        timeout: int,
        step: int,
        compressed: bool,
        staging: bool,
        encoding: bool,
    ) -> PropensityPredictResult:
        """Upload and score a single batch slice against the API.

        Called internally by :meth:`predict` for each chunk of the input
        DataFrame. Handles its own encoding, buffer lifecycle, and decoding.

        :param df: Batch slice of the input DataFrame.
        :param config: Prediction configuration matching the training variable schema.
        :param model_id: ID of the previously trained model.
        :param client_id: Client/tenant identifier.
        :param buffer_batch_size: Number of rows per buffer upload batch.
        :param verbose: Enable verbose logging.
        :param timeout: Maximum seconds to wait for the job to complete.
        :param step: Polling interval in seconds.
        :param compressed: Whether to compress the buffer payload.
        :param staging: Whether to use the staging buffer area.
        :param encoding: Whether to encode/decode field names and values.
        :return: Prediction result for this batch slice.
        :rtype: PropensityPredictResult
        :raises AnalyzrError: If encoding keys are missing or the buffer save fails.
        """
        request_id = self._get_request_id()
        fref: dict[str, Any] = {}

        if encoding:
            keys = self._keys_load(model_id=model_id or "", verbose=verbose)
            if keys is None:
                log.error("Keys not found for model_id=%s", model_id)
                raise AnalyzrError("Keys not found", detail=f"model_id={model_id}")
            data, xref, zref, rref, fref, _fref_exp, bref = self._encode(
                df,
                keys=keys,
                categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars,
                bool_vars=config.bool_vars,
                record_id_var=config.idx_var,
                verbose=verbose,
            )
        else:
            data = deepcopy(df)
            xref, zref, rref, bref = {}, {}, {}, {}

        res = self._buffer.save(
            data,
            client_id=client_id,
            request_id=request_id,
            verbose=verbose,
            batch_size=buffer_batch_size,
            compressed=compressed,
            staging=staging,
        )

        data2: pd.DataFrame | None = None
        if res["batches_saved"] == res["total_batches"]:
            self._client.post(
                self._uri,
                {
                    "command": "propensity-predict",
                    "model_id": model_id,
                    "request_id": request_id,
                    "client_id": client_id,
                    "idx_field": fref["forward"][config.idx_var]
                    if encoding and config.idx_var
                    else config.idx_var,
                    "categorical_fields": [
                        fref["forward"][v] for v in config.categorical_vars
                    ]
                    if encoding
                    else config.categorical_vars,
                    "staging": staging,
                },
            )
            self._poller.poll(
                payload={
                    "request_id": request_id,
                    "client_id": client_id,
                    "command": "task-status",
                },
                timeout=timeout,
                step=step,
                verbose=verbose,
            )
            data2 = self._buffer.read(
                request_id=request_id,
                client_id=client_id,
                dataframe_name="res",
                verbose=verbose,
                staging=True,
            )

        self._buffer.clear(request_id=request_id, client_id=client_id, verbose=verbose)

        if encoding and data2 is not None:
            data2 = self._decode(
                data2,
                categorical_vars=config.categorical_vars,
                numerical_vars=config.numerical_vars,
                bool_vars=config.bool_vars,
                record_id_var=config.idx_var,
                xref=xref,
                zref=zref,
                rref=rref,
                fref=fref,
                bref=bref,
                verbose=verbose,
            )

        return PropensityPredictResult(model_id=model_id or "", data=data2)
