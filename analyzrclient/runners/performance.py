"""Runner for the performance analysis pipeline (train, run, hierarchy retrieval, purge)."""

from __future__ import annotations

import logging
import time
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
    """Runs the performance analysis pipeline (train, run, hierarchy, purge).

    :param client: Authenticated SAML SSO client for API communication.
    :param base_url: Root URL of the Analyzr API tenant.
    """

    _uri: str
    _perf_codec: PerformanceCodec

    def __init__(self, client: Any = None, base_url: str | None = None) -> None:
        super().__init__(client=client, base_url=base_url, codec=PerformanceCodec())
        self._perf_codec = PerformanceCodec()
        self._uri = f"{self._base_url}/analytics/"

    def train(
        self,
        df: pd.DataFrame | None,
        config: PerformanceModelConfig,
        client_id: str | None = None,
        buffer_batch_size: int = 1000,
        verbose: bool = False,
        timeout: int = 600,
        step: int = 2,
        poll: bool = True,
        compressed: bool = False,
        staging: bool = True,
        encoding: bool = True,
        store_type: str = "pandas",
        database_config: dict[str, Any] | None = None,
        auto_analyze: bool | dict[str, Any] = False,
        analysis_depth: int = 1,
        fiscal_year_start_month: int = 1,
        forecasting_enabled: bool = False,
        targets: dict[str, str] | None = None,
        precompute_targets: bool | dict[str, Any] = False,
    ) -> PerformanceTrainResult:
        """Train a performance analysis model and optionally poll for results.

        For cloud data sources (Snowflake, Databricks), set ``store_type='trino'``
        and provide ``database_config`` with connection details. The API will ingest
        the source data into Parquet/Iceberg and use Trino for analysis queries.
        In this mode, ``df`` should be ``None`` — data originates from the external source.

        :param df: Input DataFrame; pass ``None`` when using a cloud store type.
        :param config: Model configuration including variable definitions and hierarchy specs.
        :param client_id: Client/tenant identifier.
        :param buffer_batch_size: Number of rows per buffer upload batch.
        :param verbose: Enable verbose logging.
        :param timeout: Maximum seconds to wait for the job to complete.
        :param step: Polling interval in seconds.
        :param poll: Whether to block and poll until the job finishes.
        :param compressed: Whether to compress the buffer payload.
        :param staging: Whether to use the staging buffer area.
        :param encoding: Whether to encode field names and values before upload.
        :param store_type: Data backend — ``'pandas'`` for in-memory or ``'trino'`` for cloud.
        :param database_config: Connection details for cloud store types (Snowflake/Databricks).
        :param auto_analyze: When ``True`` or ``{'enabled': True}``, triggers post-training
                             cache warming across all hierarchy addresses.
        :param analysis_depth: Depth of auto-analysis traversal (1 = top level only).
                               Only meaningful when ``auto_analyze`` is enabled.
        :param fiscal_year_start_month: Calendar month (1–12) on which the fiscal year begins.
                                        Defaults to ``1`` (January).
        :param forecasting_enabled: Enable forecasting during training when ``True``.
        :param targets: Mapping of measure names to their target field names,
                        e.g. ``{'revenue': 'revenue_target', 'gross_additions': 'gross_additions_target'}``.
                        When provided, target comparisons become available on the trained model.
        :param precompute_targets: When ``True`` or ``{'enabled': True}``, computes all target
                                   scenarios for every hierarchy address during training rather
                                   than deferring to run-time. Requires ``targets`` to be set.
        :return: Training result containing the model ID and analysis output.
        :rtype: PerformanceTrainResult
        :raises AnalyzrError: If the DataFrame is missing for pandas mode, the buffer
                              save fails, or the codec is not configured.
        """
        request_id = self._get_request_id()
        if verbose:
            log.info("Model ID: %s", request_id)

        is_cloud = store_type != "pandas"
        fref: dict[str, Any] = {}
        xref: dict[str, Any] = {}

        if is_cloud and encoding:
            log.warning(
                "Encoding is not supported for cloud databases. Setting encoding=False."
            )
            encoding = False

        if not is_cloud:
            if df is None:
                raise AnalyzrError("DataFrame is required when store_type is pandas")
            if encoding:
                data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                    df,
                    categorical_vars=config.dimensional_vars,
                    numerical_vars=config.primary_vars,
                    bool_vars=[],
                    record_id_var=config.idx_var,
                    verbose=verbose,
                    numerical=False,
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
                data = df

            res = self._buffer.save(
                data,
                client_id=client_id,
                request_id=request_id,
                verbose=verbose,
                batch_size=buffer_batch_size,
                compressed=compressed,
                staging=staging,
            )
            if res["batches_saved"] != res["total_batches"]:
                log.error("Buffer save failed: %s", res)
                raise AnalyzrError(
                    "Buffer save failed",
                    detail=f"request_id={request_id}, response={res}",
                )

        analysis: dict[str, Any] = {}

        train_payload: dict[str, Any] = {
            "command": "analyze-train",
            "request_id": request_id,
            "job_id": request_id,
            "client_id": client_id,
            "idx_field": fref["forward"][config.idx_var]
            if encoding and config.idx_var
            else config.idx_var,
            "time_field": fref["forward"][config.time_var]
            if encoding
            else config.time_var,
            "outcome_var": fref["forward"][config.outcome_var]
            if encoding
            else config.outcome_var,
            "primary_fields": [fref["forward"][v] for v in config.primary_vars]
            if encoding
            else config.primary_vars,
            "dimensional_fields": [fref["forward"][v] for v in config.dimensional_vars]
            if encoding
            else config.dimensional_vars,
            "edges": self._perf_codec.encode_edges(config.edges, fref)
            if encoding
            else config.edges,
            "hierarchies": self._perf_codec.encode_hierarchies(config.hierarchies, fref)
            if encoding
            else config.hierarchies,
            "udf": self._perf_codec.encode_udf(config.udf, fref)
            if encoding
            else config.udf,
            "coef": self._perf_codec.encode_coefs(config.coef, fref)
            if encoding
            else config.coef,
            "staging": staging,
            "store_type": store_type,
            "auto_analyze": {"enabled": True} if auto_analyze is True else auto_analyze,
            "analysis_depth": analysis_depth,
            "fiscal_year_start_month": fiscal_year_start_month,
            "forecasting_enabled": forecasting_enabled,
            "targets": targets or {},
            "precompute_targets": {"enabled": True} if precompute_targets is True else precompute_targets,
        }
        if database_config is not None:
            train_payload["database_config"] = database_config

        self._client.post(self._uri, train_payload)

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
                read_config = PerformanceReadConfig(
                    outcome_var=fref["forward"][config.outcome_var]
                    if encoding
                    else config.outcome_var,
                )
                raw_analysis = self._read_performance_analysis(
                    request_id=request_id,
                    client_id=client_id,
                    read_config=read_config,
                )
                if self._codec is None:
                    log.error("No codec configured for performance runner")
                    raise AnalyzrError("No codec configured for performance runner")
                encode_keys = {
                    "fref": fref if encoding else {},
                    "xref": xref if encoding else {},
                }
                result = self._codec.decode_train_results(
                    {},
                    encode_keys,
                    config,
                    request_id,
                    encoding,
                    raw_analysis=raw_analysis,
                )
                analysis = result.analysis
            else:
                log.warning(
                    "Training returned status: %s",
                    res2.get("response", {}).get("status"),
                )

        if poll and not is_cloud:
            self._buffer.clear(
                request_id=request_id, client_id=client_id, verbose=verbose
            )

        return PerformanceTrainResult(model_id=request_id, analysis=analysis)

    def run(
        self,
        df: pd.DataFrame | None,
        model_id: str,
        client_id: str,
        run_config: PerformanceRunConfig,
        config: PerformanceModelConfig | None = None,
        buffer_batch_size: int = 1000,
        verbose: bool = False,
        timeout: int = 600,
        step: int = 2,
        compressed: bool = False,
        staging: bool = True,
        encoding: bool = True,
    ) -> PerformanceRunResult:
        """Execute inference on a pre-trained performance model, optionally refreshing data.

        :param df: Optional refreshed DataFrame; when provided, replaces the original
                   training data for this run.
        :param model_id: ID of the previously trained model.
        :param client_id: Client/tenant identifier.
        :param run_config: Run-time configuration (address, outcome var, period, fiscal context).
        :param config: Original model configuration; required to resolve field mappings.
        :param buffer_batch_size: Number of rows per buffer upload batch.
        :param verbose: Enable verbose logging.
        :param timeout: Maximum seconds to wait for the job to complete.
        :param step: Polling interval in seconds.
        :param compressed: Whether to compress the buffer payload.
        :param staging: Whether to use the staging buffer area.
        :param encoding: Whether to encode field names and values.
        :return: Run result containing the request ID, model ID, and analysis output.
        :rtype: PerformanceRunResult
        :raises AnalyzrError: If ``config`` is ``None``, encoding keys are missing,
                              the buffer save fails, or the codec is not configured.
        """
        if config is None:
            log.error("PerformanceModelConfig is required for run()")
            raise AnalyzrError(
                "PerformanceModelConfig is required for run()",
                detail=f"model_id={model_id}",
            )

        request_id = self._get_request_id()
        refresh_data = df is not None and not df.empty
        fref: dict[str, Any] = {}
        xref: dict[str, Any] = {}

        if refresh_data:
            df_non_null = cast(pd.DataFrame, df)
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    log.error("Keys not found for model_id=%s", model_id)
                    raise AnalyzrError("Keys not found", detail=f"model_id={model_id}")
                data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                    df_non_null,
                    keys=keys,
                    categorical_vars=config.dimensional_vars,
                    numerical_vars=config.primary_vars,
                    bool_vars=[],
                    record_id_var=config.idx_var,
                    verbose=verbose,
                    numerical=False,
                )
            else:
                data = df_non_null

            res = self._buffer.save(
                data,
                client_id=client_id,
                request_id=request_id,
                verbose=verbose,
                batch_size=buffer_batch_size,
                compressed=compressed,
                staging=staging,
            )
            if res["batches_saved"] != res["total_batches"]:
                log.error("Buffer save failed: %s", res)
                raise AnalyzrError(
                    "Buffer save failed",
                    detail=f"request_id={request_id}, response={res}",
                )
        else:
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    log.error("Keys not found for model_id=%s", model_id)
                    raise AnalyzrError("Keys not found", detail=f"model_id={model_id}")
                fref = keys["fref"]
                xref = keys["xref"]

        address = (
            self._perf_codec.encode_address(
                list(run_config.address.values()),
                config.dimensional_vars,
                fref,
                xref,
            )
            if encoding
            else tuple(run_config.address.values())
        )

        encoded_outcome = (
            fref["forward"][run_config.outcome_var]
            if encoding
            else run_config.outcome_var
        )

        run_payload: dict[str, Any] = {
            "command": "analyze-run",
            "request_id": request_id,
            "model_id": model_id,
            "client_id": client_id,
            "idx_field": fref["forward"][config.idx_var]
            if encoding and config.idx_var
            else config.idx_var,
            "time_field": fref["forward"][config.time_var]
            if encoding
            else config.time_var,
            "outcome_var": encoded_outcome,
            "address": address,
            "main_stat": run_config.main_stat,
            "primary_fields": [fref["forward"][v] for v in config.primary_vars]
            if encoding
            else config.primary_vars,
            "dimensional_fields": [fref["forward"][v] for v in config.dimensional_vars]
            if encoding
            else config.dimensional_vars,
            "staging": staging,
            "refresh_data": refresh_data,
        }
        if run_config.period:
            run_payload["period"] = run_config.period
        if run_config.fiscal:
            run_payload.update(run_config.fiscal.to_payload())

        self._client.post(self._uri, run_payload)

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

        analysis: dict[str, Any] = {}
        if res2.get("response", {}).get("status") == "Complete":
            read_config = PerformanceReadConfig(
                outcome_var=encoded_outcome,
                main_stat=run_config.main_stat,
                address=address,
                period=run_config.period,
                fiscal=run_config.fiscal,
            )
            raw_analysis = self._read_performance_analysis(
                request_id=model_id,
                client_id=client_id,
                read_config=read_config,
            )
            if self._codec is None:
                log.error("No codec configured for performance runner")
                raise AnalyzrError("No codec configured for performance runner")
            encode_keys = {
                "fref": fref if encoding else {},
                "xref": xref if encoding else {},
            }
            result = self._codec.decode_train_results(
                {},
                encode_keys,
                config,
                request_id,
                encoding,
                raw_analysis=raw_analysis,
            )
            analysis = result.analysis
        else:
            log.warning(
                "Run returned status: %s", res2.get("response", {}).get("status")
            )

        if refresh_data:
            self._buffer.clear(
                request_id=request_id, client_id=client_id, verbose=verbose
            )

        return PerformanceRunResult(
            request_id=request_id,
            model_id=model_id,
            analysis=analysis,
        )

    def _read_performance_analysis(
        self,
        request_id: str,
        client_id: str | None,
        read_config: PerformanceReadConfig,
    ) -> dict[str, Any]:
        """Retrieve performance analysis results from the API result cache.

        :param request_id: Completed job/model ID whose results should be fetched.
        :param client_id: Client/tenant identifier.
        :param read_config: Parameters controlling which analysis slice to retrieve
                            (outcome var, address, period, main stat, fiscal context).
        :return: Raw analysis dict from the API response.
        :rtype: dict[str, Any]
        :raises AnalyzrError: If the API returns a non-200 status or an empty response.
        """
        payload: dict[str, Any] = {
            "command": "read-performance-analysis",
            "request_id": request_id,
            "client_id": client_id,
            "outcome_var": read_config.outcome_var,
            "address": list(read_config.address.values())
            if isinstance(read_config.address, dict)
            else read_config.address,
            "period": read_config.period,
            "main_stat": read_config.main_stat,
        }
        if read_config.fiscal:
            payload.update(read_config.fiscal.to_payload())

        res = self._client.post(self._uri, payload)
        if res.get("status") == 200 and res.get("response") is not None:
            return res["response"]
        log.error("Failed to read performance analysis: %s", res)
        raise AnalyzrError(
            "Failed to read performance analysis",
            detail=f"request_id={request_id}, response={res}",
        )

    def get_hierarchy_results(
        self,
        model_id: str,
        client_id: str,
        hierarchy_name: str,
        outcome_var: str,
        address: dict[str, Any] | None = None,
        hierarchy_type: str = "dimensional",
        main_stat: str = "prior_year_delta",
        period: str | None = None,
        fiscal_context: dict[str, Any] | None = None,
        store_type: str = "pandas",
        verbose: bool = False,
        timeout: int = 600,
        step: int = 2,
    ) -> dict[str, Any]:
        """Retrieve hierarchy results for a single dimension at a given address.

        Mirrors the frontend's per-hierarchy call in the dimensional graph component:
        one call per hierarchy name at the current drill-down address. Submits an
        async task, polls until complete, then reads results back via the API's
        read-hierarchy-results command (cache key computation stays server-side).

        :param model_id: Trained model ID (also serves as job_id).
        :param client_id: Client/tenant identifier.
        :param hierarchy_name: Root dimension column name of the hierarchy to retrieve
                               (e.g. ``'state'``, ``'product'``). Must match the ``dimension``
                               field in the hierarchy config, not the display ``name``.
        :param outcome_var: Measure to analyse (e.g. ``'net_additions'``).
        :param address: Dimensional address dict; ``None`` values indicate the top level.
        :param hierarchy_type: ``'dimensional'`` or ``'accounting'``.
        :param main_stat: Comparison statistic (default ``'prior_year_delta'``).
        :param period: Time period string (e.g. ``'2024-01'``).
        :param fiscal_context: Optional dict with fiscal fields (``fiscal_year``,
                               ``fiscal_quarter``, ``fiscal_period``, ``fiscal_week``, etc.).
        :param store_type: Data backend used during training — ``'pandas'`` or ``'trino'``.
        :param verbose: Enable verbose logging.
        :param timeout: Polling timeout in seconds.
        :param step: Polling interval in seconds.
        :return: Raw hierarchy results dict from the API.
        :rtype: dict[str, Any]
        :raises AnalyzrError: If the task fails or results are not found after polling completes.
        """
        request_id = self._get_request_id()
        t_start = time.perf_counter()

        payload: dict[str, Any] = {
            "command": "analyze-get-hierarchy-results",
            "request_id": request_id,
            "model_id": model_id,
            "job_id": model_id,
            "client_id": client_id,
            "hierarchy_name": hierarchy_name,
            "hierarchy_type": hierarchy_type,
            "outcome_var": outcome_var,
            "address": list(address.values()) if address else None,
            "main_stat": main_stat,
            "store_type": store_type,
        }
        if period is not None:
            payload["period"] = period
        if fiscal_context:
            payload.update(fiscal_context)

        t_submit = time.perf_counter()
        self._client.post(self._uri, payload)
        t_submitted = time.perf_counter()
        if verbose:
            log.info("[hierarchy:%s] submitted in %.2fs", hierarchy_name, t_submitted - t_submit)

        t_poll_start = time.perf_counter()
        res = self._poller.poll(
            payload={
                "request_id": request_id,
                "client_id": client_id,
                "command": "task-status",
            },
            timeout=timeout,
            step=step,
            verbose=verbose,
        )
        t_poll_end = time.perf_counter()
        if verbose:
            log.info("[hierarchy:%s] queue+execution: %.2fs", hierarchy_name, t_poll_end - t_poll_start)

        status = res.get("response", {}).get("status", "")
        if status != "Complete":
            log.warning("Hierarchy results returned status: %s", status)
            raise AnalyzrError(
                "Hierarchy results task did not complete",
                detail=f"model_id={model_id}, hierarchy_name={hierarchy_name}, status={status}",
            )

        read_payload: dict[str, Any] = {
            "command": "read-hierarchy-results",
            "request_id": request_id,
            "job_id": model_id,
            "client_id": client_id,
            "hierarchy_name": hierarchy_name,
            "hierarchy_type": hierarchy_type,
            "outcome_var": outcome_var,
            "address": list(address.values()) if address else None,
            "main_stat": main_stat,
        }
        if period is not None:
            read_payload["period"] = period
        if fiscal_context:
            read_payload.update(fiscal_context)

        t_read_start = time.perf_counter()
        read_res = self._client.post(self._uri, read_payload)
        t_read_end = time.perf_counter()

        if verbose:
            log.info(
                "[hierarchy:%s] read-back: %.2fs | total: %.2fs",
                hierarchy_name,
                t_read_end - t_read_start,
                t_read_end - t_start,
            )

        if read_res.get("status") == 200 and read_res.get("response") is not None:
            return read_res["response"]

        log.error("Failed to read hierarchy results: %s", read_res)
        raise AnalyzrError(
            "Failed to read hierarchy results",
            detail=f"model_id={model_id}, hierarchy_name={hierarchy_name}, response={read_res}",
        )

    def purge(
        self,
        model_id: str | None = None,
        client_id: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Remove all stored data associated with a performance analysis model.

        :param model_id: ID of the model whose data should be purged.
        :param client_id: Client/tenant identifier.
        :param verbose: Enable verbose logging.
        :return: API response dict confirming the purge.
        :rtype: dict[str, Any]
        :raises AnalyzrError: If the API returns a non-200 status.
        """
        if verbose:
            log.info("Purging performance model data...")
        res = self._client.post(
            self._uri,
            {
                "command": "analyze-purge",
                "model_id": model_id,
                "client_id": client_id,
            },
        )
        if res["status"] != 200:
            log.error("Could not purge model: %s", res)
            raise AnalyzrError(
                "Could not purge model", detail=f"model_id={model_id}, response={res}"
            )
        return res["response"]

