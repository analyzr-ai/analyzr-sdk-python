"""Custom exception hierarchy for the Analyzr SDK."""

from __future__ import annotations


class AnalyzrError(Exception):
    """Base exception for all Analyzr SDK errors.

    Subclass this for domain-specific error categories as needed.
    All SDK code should raise ``AnalyzrError`` (or a subclass) instead of
    printing error messages to stdout.

    :param message: Primary human-readable description of the error.
    :param detail: Optional supplementary detail (e.g. request IDs, raw responses).
    """

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self._format())

    def _format(self) -> str:
        """Build the final exception string, appending detail on a separate line when present.

        :return: Formatted exception message string.
        :rtype: str
        """
        parts = [f"[Analyzr] {self.message}"]
        if self.detail:
            parts.append(f"  Detail: {self.detail}")
        return "\n".join(parts)

    @property
    def user_message(self) -> str:
        """Clean, user-facing error string suitable for display in notebooks or logs."""
        return self.message
