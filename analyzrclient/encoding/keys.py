"""Disk-based persistence for encoding key maps.

Serializes and deserializes the full set of encoding keys (xref, zref, fref,
rref, bref, fref_exp) produced during the encode phase so that they can be
reused for subsequent prediction or scoring runs without recomputing them.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

from ..constants import TEMP_DIR


class KeyStore:
    """Persists encoding keys (xref, zref, fref, etc.) to disk."""

    @staticmethod
    def save(model_id: str, keys: dict[str, Any]) -> None:
        """Persist encoding keys to a binary file under the configured temp directory.

        Creates the temp directory if it does not already exist.

        :param model_id: Unique model identifier used as the filename stem.
        :param keys: Encoding keys dict to serialize (xref, zref, fref, etc.).
        """
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        filename = f"{TEMP_DIR}/{model_id}.bin"
        with open(filename, "wb") as f:
            pickle.dump(keys, f)

    @staticmethod
    def load(model_id: str) -> dict[str, Any] | None:
        """Load persisted encoding keys from disk for a given model.

        :param model_id: Unique model identifier used as the filename stem.
        :return: Deserialized encoding keys dict, or ``None`` if no file exists.
        :rtype: dict or None
        """
        try:
            filename = f"{TEMP_DIR}/{model_id}.bin"
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
