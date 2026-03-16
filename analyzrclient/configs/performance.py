from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FiscalConfig:
    """Fiscal calendar configuration for performance analysis."""

    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    fiscal_period: int | None = None
    fiscal_week: int | None = None
    fiscal_day_of_period: int | None = None
    fiscal_day_of_week: int | None = None
    fiscal_year_start_month: int = 1
    fiscal_calendar_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return non-None fiscal fields for API payloads."""
        return {k: v for k, v in {
            'fiscal_year': self.fiscal_year,
            'fiscal_quarter': self.fiscal_quarter,
            'fiscal_period': self.fiscal_period,
            'fiscal_week': self.fiscal_week,
            'fiscal_day_of_period': self.fiscal_day_of_period,
            'fiscal_day_of_week': self.fiscal_day_of_week,
            'fiscal_year_start_month': self.fiscal_year_start_month,
            'fiscal_calendar_id': self.fiscal_calendar_id,
        }.items() if v is not None}

    @property
    def is_fiscal(self) -> bool:
        return self.fiscal_calendar_id is not None


@dataclass
class PerformanceModelConfig:
    """Configuration for performance analysis model training."""

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
    """Advanced configuration for performance analysis."""

    pass


@dataclass
class PerformanceReadConfig:
    """Configuration for reading cached performance analysis results.

    Identifies a specific analysis result by measure, address, period,
    and comparison strategy. The API resolves the correct cache entry
    from these parameters.

    Valid main_stat values (comparison strategies):
        Point-in-time: prior_day_delta, prior_week_delta, prior_month_delta,
            prior_quarter_delta, prior_year_delta
        Period-to-date: week_to_date, month_to_date, quarter_to_date, year_to_date
        Sequential: prior_week_to_date_delta, prior_month_to_date_delta,
            prior_quarter_to_date_delta
        Year-over-year: prior_year_week_to_date_delta, prior_year_month_to_date_delta,
            prior_year_quarter_to_date_delta, prior_year_to_date_delta
        Target: target_delta, target_week_to_date_delta, target_month_to_date_delta,
            target_quarter_to_date_delta, target_year_to_date_delta
        Fiscal: fiscal_week_vs_prior_year_delta, fiscal_period_vs_prior_year_delta,
            fiscal_quarter_vs_prior_year_delta, fiscal_year_vs_prior_year_delta, etc.
    """

    outcome_var: str
    main_stat: str = 'prior_year_delta'
    address: Any = None  # dict, list, or tuple depending on encoding path
    period: str | None = None
    fiscal: FiscalConfig | None = None


@dataclass
class PerformanceRunConfig:
    """Configuration for running a pre-trained performance model."""

    outcome_var: str
    main_stat: str = 'prior_year_delta'
    address: dict[str, str | None] = field(default_factory=dict)
    period: str | None = None
    fiscal: FiscalConfig | None = None
