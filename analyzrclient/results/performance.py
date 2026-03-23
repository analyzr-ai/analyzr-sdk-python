"""Result dataclasses for performance analysis workflows.

Provides ``PerformanceTrainResult`` and ``PerformanceRunResult`` which
encapsulate the structured outputs returned by the analyzr analytics API
after performance model training and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceTrainResult:
    """Result from training a performance analysis model.

    Holds the opaque model identifier assigned by the analytics engine and
    the full analysis payload returned after training completes.

    :param model_id: Unique identifier for the trained model, used to
        reference it in subsequent run or read requests.
    :param analysis: Raw analysis output dict returned by the API.  Structure
        is determined by the model type and may include diagnostic statistics,
        node summaries, and hierarchy rollups.
    """

    model_id: str
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``model_id`` and ``analysis``.
        :rtype: dict[str, Any]
        """
        return {"model_id": self.model_id, "analysis": self.analysis}


@dataclass
class PerformanceRunResult:
    """Result from running a pre-trained performance model.

    Holds the request identifier, model identifier, and the analysis payload
    returned by the analytics engine for a specific run invocation.

    :param request_id: Unique identifier for this specific run request,
        useful for correlating async job completions.
    :param model_id: Unique identifier of the model that was executed.
    :param analysis: Raw analysis output dict returned by the API.  Structure
        mirrors the training result but is scoped to the run's address,
        period, and comparison strategy.
    """

    request_id: str
    model_id: str
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary.

        :return: Dict containing ``request_id``, ``model_id``, and
            ``analysis``.
        :rtype: dict[str, Any]
        """
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "analysis": self.analysis,
        }
