"""Abstract base class defining the codec contract for analytics domain results.

Each analytics domain (cluster, regression, propensity, etc.) provides a concrete
subclass that knows how to map raw buffer DataFrames back to typed result objects.
"""

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
        """Return buffer dataframe names needed to decode train results.

        :param config: Domain-specific training configuration object.
        :return: List of buffer frame name strings to fetch before decoding.
        :rtype: list[str]
        """
        ...

    @abstractmethod
    def decode_train_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> Any:
        """Decode raw buffer frames into a typed train result.

        :param frames: Mapping of frame name to DataFrame fetched from the buffer.
        :param keys: Encoding keys (xref, zref, fref, etc.) from the encode phase.
        :param config: Domain-specific training configuration object.
        :param model_id: Unique identifier for the trained model.
        :param encoding: Whether field-name and variable encoding was applied during training.
        :return: Domain-specific typed result object.
        """
        ...

    def get_predict_frame_names(self, config: Any) -> list[str]:
        """Return buffer dataframe names needed to decode predict results.

        :param config: Domain-specific training configuration object.
        :return: List of buffer frame name strings to fetch before decoding.
        :rtype: list[str]
        :raises NotImplementedError: When the domain does not support predict.
        """
        raise NotImplementedError

    def decode_predict_results(
        self,
        frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> Any:
        """Decode raw buffer frames into a typed predict result.

        :param frames: Mapping of frame name to DataFrame fetched from the buffer.
        :param keys: Encoding keys (xref, zref, fref, etc.) from the encode phase.
        :param config: Domain-specific training configuration object.
        :param model_id: Unique identifier for the model used during prediction.
        :param encoding: Whether field-name and variable encoding was applied.
        :return: Domain-specific typed predict result object.
        :raises NotImplementedError: When the domain does not support predict.
        """
        raise NotImplementedError
