"""Human-readable, redacted task digest formatting.

This module is intentionally schema-independent groundwork: callers pass plain
safe marker-like dictionaries and receive operator prose suitable for context
injection. It does not read or write gateway/kanban storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agent.redact import redact_sensitive_text


_DEFAULT_SUMMARY = "제공된 요약 없음 / no summary provided."
_DEFAULT_PROGRESS = "확인된 진행 항목 없음 / no confirmed progress items."
_DEFAULT_NEXT_ACTION = "확인된 다음 action 없음 / no confirmed next action."
_DEFAULT_UNCERTAINTY = "세부 정보가 제공되지 않아 불확실함 / details were not provided, so this remains uncertain."

_PRIVATE_KEYS = frozenset(
    {
        "raw_payload",
        "payload",
        "private_payload",
        "private_notes",
        "private_note",
        "raw",
        "transcript",
        "messages",
        "history",
        "secret",
        "secrets",
        "token",
        "tokens",
        "api_key",
        "apikey",
        "password",
        "authorization",
        "credentials",
    }
)


@dataclass(frozen=True)
class RedactedTaskDigest:
    """Small schema-free task digest for safe context injection."""

    title: str
    summary: str = _DEFAULT_SUMMARY
    progress: tuple[str, ...] = field(default_factory=lambda: (_DEFAULT_PROGRESS,))
    next_actions: tuple[str, ...] = field(default_factory=lambda: (_DEFAULT_NEXT_ACTION,))
    uncertainties: tuple[str, ...] = field(default_factory=lambda: (_DEFAULT_UNCERTAINTY,))

    def format_for_context(self) -> str:
        """Render neutral Korean/English operator prose, not raw JSON."""
        return format_task_digest_context(self)


def build_task_digest(data: dict[str, Any] | None = None, **overrides: Any) -> RedactedTaskDigest:
    """Build a redacted digest from plain safe marker-like input data.

    The input is deliberately plain data so this slice stays independent from
    migrations or gateway wiring. Private/raw fields are ignored entirely.
    """
    source: dict[str, Any] = {}
    if data:
        source.update(data)
    source.update(overrides)

    title = _first_text(
        source,
        ("safe_title", "safe_task_title", "safe_session_title", "title", "task_title", "session_title"),
        default="Untitled task / 제목 없음",
    )
    summary = _first_text(
        source,
        ("safe_summary", "safe_description", "summary", "description"),
        default=_DEFAULT_SUMMARY,
    )
    progress = _text_tuple(
        _first_value(source, ("safe_progress", "safe_progress_items", "progress", "progress_items")),
        default=_DEFAULT_PROGRESS,
    )
    next_actions = _text_tuple(
        _first_value(source, ("safe_next_actions", "safe_next_action", "next_actions", "next_action")),
        default=_DEFAULT_NEXT_ACTION,
    )
    uncertainties = _text_tuple(
        _first_value(source, ("safe_uncertainties", "safe_uncertainty", "uncertainties", "uncertainty")),
        default=_DEFAULT_UNCERTAINTY,
    )

    return RedactedTaskDigest(
        title=_clean_text(title),
        summary=_clean_text(summary),
        progress=tuple(_clean_text(item) for item in progress),
        next_actions=tuple(_clean_text(item) for item in next_actions),
        uncertainties=tuple(_clean_text(item) for item in uncertainties),
    )


def format_task_digest_context(digest: RedactedTaskDigest | dict[str, Any]) -> str:
    """Format a digest as compact human-readable context prose."""
    if not isinstance(digest, RedactedTaskDigest):
        digest = build_task_digest(digest)

    lines = [
        "RedactedTaskDigest:",
        f"작업: {digest.title}",
        f"요약: {digest.summary}",
        "진행:",
    ]
    lines.extend(f"- {item}" for item in digest.progress)
    lines.append("다음:")
    lines.extend(f"- {item}" for item in digest.next_actions)
    lines.append("불확실:")
    lines.extend(f"- {item}" for item in digest.uncertainties)
    return "\n".join(_clean_text(line) for line in lines)


def _first_value(source: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, "", [], ()):  # keep falsey booleans out of prose
            return source[key]
    return None


def _first_text(source: dict[str, Any], keys: Iterable[str], *, default: str) -> str:
    value = _first_value(source, keys)
    text_items = _text_tuple(value, default=default)
    if not text_items:
        return default
    return "; ".join(text_items)


def _text_tuple(value: Any, *, default: str) -> tuple[str, ...]:
    items = tuple(item for item in _iter_text(value) if item)
    return items or (default,)


def _iter_text(value: Any) -> Iterable[str]:
    if value in (None, "", [], ()):  # type: ignore[comparison-overlap]
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)
        return
    if isinstance(value, dict):
        for key in ("text", "summary", "label", "title", "value", "confidence"):
            if key in value and key.lower() not in _PRIVATE_KEYS:
                yield from _iter_text(value[key])
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_text(item)
        return
    yield str(value)


def _clean_text(text: Any) -> str:
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", " ")
    for private_key in _PRIVATE_KEYS:
        text = text.replace(private_key, private_key.replace("_", " "))
    text = " ".join(text.split())
    return redact_sensitive_text(text, force=True)
