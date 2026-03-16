from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DomainCodec(ABC):
    """Base class for domain-specific result encoding/decoding.

    Each analytics domain (cluster, regression, etc.) subclasses this.
    The runner delegates all decode logic to its codec instance.
    """

    @abstractmethod
    def get_train_frame_names(self, config: Any) -> list[str]:
        """Return buffer dataframe names needed to decode train results."""
        ...

    @abstractmethod
    def decode_train_results(
        self, frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> Any:
        """Decode raw buffer frames into a typed train result."""
        ...

    def get_predict_frame_names(self, config: Any) -> list[str]:
        """Return buffer dataframe names needed to decode predict results."""
        raise NotImplementedError

    def decode_predict_results(
        self, frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> Any:
        """Decode raw buffer frames into a typed predict result."""
        raise NotImplementedError
