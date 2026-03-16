from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceTrainResult:
    """Result from training a performance analysis model."""

    model_id: str
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {'model_id': self.model_id, 'analysis': self.analysis}


@dataclass
class PerformanceRunResult:
    """Result from running a pre-trained performance model."""

    request_id: str
    model_id: str
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'model_id': self.model_id,
            'analysis': self.analysis,
        }
