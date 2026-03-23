"""Configuration dataclasses for performance analysis workflows.

Provides ``FiscalConfig``, ``PerformanceModelConfig``, ``PerformanceAdvancedConfig``,
``PerformanceReadConfig``, and ``PerformanceRunConfig`` used to parameterize
performance model training, execution, and cached-result retrieval via the
analyzr API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FiscalConfig:
    """Fiscal calendar configuration for performance analysis.

    Encapsulates all fiscal-period identifiers needed to resolve a specific
    point or range on a custom fiscal calendar.  When ``fiscal_calendar_id``
    is set the instance is considered fiscal (see :attr:`is_fiscal`).

    :param fiscal_year: Fiscal year number, or ``None`` to omit.
    :param fiscal_quarter: Fiscal quarter (1–4), or ``None`` to omit.
    :param fiscal_period: Fiscal period number within the year, or ``None`` to omit.
    :param fiscal_week: Fiscal week number, or ``None`` to omit.
    :param fiscal_day_of_period: Day index within the fiscal period, or ``None`` to omit.
    :param fiscal_day_of_week: Day index within the fiscal week, or ``None`` to omit.
    :param fiscal_year_start_month: Calendar month (1–12) on which the fiscal year
        begins.  Defaults to ``1`` (January).
    :param fiscal_calendar_id: Opaque identifier for a custom fiscal calendar
        registered in the platform.  When provided, enables fiscal-relative
        comparison strategies.
    """

    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    fiscal_period: int | None = None
    fiscal_week: int | None = None
    fiscal_day_of_period: int | None = None
    fiscal_day_of_week: int | None = None
    fiscal_year_start_month: int = 1
    fiscal_calendar_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a dict of non-``None`` fiscal fields suitable for API payloads.

        :return: Mapping of fiscal field names to their non-``None`` values.
        :rtype: dict[str, Any]
        """
        return {
            k: v
            for k, v in {
                "fiscal_year": self.fiscal_year,
                "fiscal_quarter": self.fiscal_quarter,
                "fiscal_period": self.fiscal_period,
                "fiscal_week": self.fiscal_week,
                "fiscal_day_of_period": self.fiscal_day_of_period,
                "fiscal_day_of_week": self.fiscal_day_of_week,
                "fiscal_year_start_month": self.fiscal_year_start_month,
                "fiscal_calendar_id": self.fiscal_calendar_id,
            }.items()
            if v is not None
        }

    @property
    def is_fiscal(self) -> bool:
        """Return ``True`` when a custom fiscal calendar is configured."""
        return self.fiscal_calendar_id is not None


@dataclass
class PerformanceModelConfig:
    """Configuration for performance analysis model training.

    Defines the structural inputs — variable roles, graph edges, and
    dimension hierarchies — required by the analytics engine to build a
    performance model.

    :param time_var: Name of the column representing the time dimension.
    :param outcome_var: Name of the target/outcome variable column.
    :param primary_vars: Variable names that form the primary measurement set.
    :param dimensional_vars: Variable names used as dimensional breakdowns.
    :param edges: Directed edges ``(source, target)`` describing the causal
        or structural graph among variables.
    :param hierarchies: List of hierarchy descriptor dicts, each specifying
        levels of a dimensional rollup.
    :param idx_var: Name of the unique record identifier column, or ``None``
        if no explicit index is present.
    :param udf: Mapping of variable names to UDF expressions applied before
        training.
    :param coef: Mapping of variable names to fixed coefficient overrides.
    """

    time_var: str
    outcome_var: str
    primary_vars: list[str]
    dimensional_vars: list[str]
    edges: list[tuple[str, str]]
    hierarchies: list[dict[str, Any]]
    idx_var: str | None = None
    udf: dict[str, str] = field(default_factory=dict)
    coef: dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceAdvancedConfig:
    """Advanced configuration options for performance analysis.

    Reserved for future advanced tuning parameters.  Currently a placeholder
    that may be extended without breaking existing call sites.
    """

    pass


@dataclass
class PerformanceReadConfig:
    """Configuration for reading cached performance analysis results.

    Identifies a specific analysis result by measure, address, period,
    and comparison strategy.  The API resolves the correct cache entry
    from these parameters.

    Valid ``main_stat`` values (comparison strategies):

    * **Point-in-time**: ``prior_day_delta``, ``prior_week_delta``,
      ``prior_month_delta``, ``prior_quarter_delta``, ``prior_year_delta``
    * **Period-to-date**: ``week_to_date``, ``month_to_date``,
      ``quarter_to_date``, ``year_to_date``
    * **Sequential**: ``prior_week_to_date_delta``,
      ``prior_month_to_date_delta``, ``prior_quarter_to_date_delta``
    * **Year-over-year**: ``prior_year_week_to_date_delta``,
      ``prior_year_month_to_date_delta``,
      ``prior_year_quarter_to_date_delta``, ``prior_year_to_date_delta``
    * **Target**: ``target_delta``, ``target_week_to_date_delta``,
      ``target_month_to_date_delta``, ``target_quarter_to_date_delta``,
      ``target_year_to_date_delta``
    * **Fiscal**: ``fiscal_week_vs_prior_year_delta``,
      ``fiscal_period_vs_prior_year_delta``,
      ``fiscal_quarter_vs_prior_year_delta``,
      ``fiscal_year_vs_prior_year_delta``, and related variants.

    :param outcome_var: Name of the outcome variable whose cached result
        should be fetched.
    :param main_stat: Comparison strategy key identifying which cached
        statistic to retrieve.  Defaults to ``'prior_year_delta'``.
    :param address: Dimensional address (dict, list, or tuple) that pins
        the result to a specific node in the dimensional hierarchy.
    :param period: ISO date string or period label scoping the result to a
        specific time point, or ``None`` to use the latest available.
    :param fiscal: Fiscal calendar parameters when retrieving a fiscal-period
        result, or ``None`` for standard calendar lookups.
    """

    outcome_var: str
    main_stat: str = "prior_year_delta"
    address: Any = None  # dict, list, or tuple depending on encoding path
    period: str | None = None
    fiscal: FiscalConfig | None = None


@dataclass
class PerformanceRunConfig:
    """Configuration for running a pre-trained performance model.

    Specifies the outcome variable, comparison strategy, dimensional address,
    and optional time/fiscal scope used when executing a model that has
    already been trained.

    :param outcome_var: Name of the outcome variable to evaluate.
    :param main_stat: Comparison strategy key.  Defaults to
        ``'prior_year_delta'``.  See :class:`PerformanceReadConfig` for the
        full list of valid values.
    :param address: Dimensional address dict mapping dimension names to their
        selected values.  Use ``None`` values to indicate an unfiltered
        dimension.
    :param period: ISO date string or period label, or ``None`` to use the
        model's default period.
    :param fiscal: Fiscal calendar parameters for fiscal-period runs, or
        ``None`` for standard calendar execution.
    """

    outcome_var: str
    main_stat: str = "prior_year_delta"
    address: dict[str, str | None] = field(default_factory=dict)
    period: str | None = None
    fiscal: FiscalConfig | None = None
