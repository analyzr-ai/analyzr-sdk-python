from __future__ import annotations

import os
import pickle
from typing import Any

from ..constants import TEMP_DIR


class KeyStore:
    """Persists encoding keys (xref, zref, fref, etc.) to disk."""

    @staticmethod
    def save(model_id: str, keys: dict[str, Any]) -> None:
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        filename = f'{TEMP_DIR}/{model_id}.bin'
        with open(filename, 'wb') as f:
            pickle.dump(keys, f)

    @staticmethod
    def load(model_id: str) -> dict[str, Any] | None:
        try:
            filename = f'{TEMP_DIR}/{model_id}.bin'
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
