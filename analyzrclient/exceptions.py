from __future__ import annotations

class AnalyzrError(Exception):
    """Base exception for all Analyzr SDK errors.

    Subclass this for domain-specific error categories as needed.
    All SDK code should raise AnalyzrError (or a subclass) instead of
    printing error messages to stdout.
    """

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [f"[Analyzr] {self.message}"]
        if self.detail:
            parts.append(f"  Detail: {self.detail}")
        return "\n".join(parts)

    @property
    def user_message(self) -> str:
        """Clean, user-facing error string suitable for display in notebooks or logs."""
        return self.message
