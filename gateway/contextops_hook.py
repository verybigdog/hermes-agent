"""ContextOps gateway hydration canary hook (one-turn, fail-closed).

Thin Hermes-side adapter that assembles a metadata-only, one-turn ContextOps
injection block through the reviewed M4 (``runtime_assembly_preview_m4``) and
M5 (``one_turn_injection_preview_m5``) contracts. ContextOps core stays
Hermes-free; Hermes owns only this client.

Safety posture (all fail-closed to "no injection"):

* Default OFF — requires ``contextops.gateway_hydration.enabled`` to be the
  literal boolean ``True``.
* Config-gated channel scope: exact channel entries (``<chat_id>`` or
  ``<platform>:<chat_id>``) and explicit platform wildcards such as
  ``discord:*`` are supported for reviewed dogfood expansion. Missing/empty
  chat ids still fail closed. The lane comes from config only, never inferred
  from message text.
* ContextOps is imported lazily, only after the config/channel gates pass.
  An optional ``repo_path`` is inserted into ``sys.path`` just for the import
  (nothing persisted, nothing vendored). Import failure → no injection.
* M4/M5 artifacts live only inside a ``tempfile.TemporaryDirectory()`` — no
  durable Hermes memory or user-profile write, and temp paths are never
  exposed in the returned block.
* The returned block is metadata-only (counts + redacted placeholders) and is
  re-scanned here: any path separator, JSON brace, secret-shaped string, or
  configured secret/path value in the rendered text → no injection.
* No Kanban create, dispatch, send, wake, or restart from this layer; M5's
  ``injected``/``model_call``/``side_effects`` flags are re-checked and any
  unexpected value fails closed.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
from pathlib import Path

_M4_MODULE = "contextops.integration.runtime_assembly_preview"
_M5_MODULE = "contextops.integration.one_turn_injection_gate"

# Fallback one-turn message fixture for direct unit calls.  The live gateway
# passes the current prepared inbound message so M4 can prove the actual turn is
# safe; this fallback keeps direct helper calls inert and deterministic.
_FIXTURE_MESSAGE = "one-turn hydration canary fixture (metadata-only)\n"

# Identifiers rendered into the block must look like plain identifiers.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

# Substrings that must never appear in the rendered injection block:
# backslashes, JSON/provider braces, URL schemes, and common secret prefixes.
_FORBIDDEN_SUBSTRINGS = (
    "\\", "{", "}", "://", "sk-", "ghp_", "xox", "AKIA",
    "Bearer ", "BEGIN ", "PRIVATE KEY", "eyJ",
)

# Path-shaped text: a token that starts with "/", a double slash, or a
# multi-segment "/a/b" run. The single "/" inside the "redacted/metadata-only"
# placeholder matches none of these.
_PATH_SHAPE_RE = re.compile(r"(?:^|[\s:=\"'(\[])/|/{2,}|/[A-Za-z0-9_.-]+/")

_REQUIRED_FIELDS = (
    "lane",
    "m3_preview_path",
    "m3_receipt_path",
    "expected_sidecar_hash",
    "expected_effect_id",
    "approval_token",
    "approver",
    "reason",
    "target_id",
)

_MAX_PLACEHOLDER_LINES = 20
_MAX_SEMANTIC_ITEMS = 8
_MAX_SEMANTIC_CHARS = 240


def _hydration_config(config) -> dict | None:
    if not isinstance(config, dict):
        return None
    section = config.get("contextops")
    if not isinstance(section, dict):
        return None
    cfg = section.get("gateway_hydration")
    return cfg if isinstance(cfg, dict) else None


def _channel_allowed(cfg: dict, *, platform: str, chat_id: str) -> bool:
    if cfg.get("enabled") is not True:
        return False
    allowed = cfg.get("allowed_channels")
    if not isinstance(allowed, list) or not allowed:
        return False
    clean_platform = str(platform or "").strip()
    clean_chat_id = str(chat_id or "").strip()
    if not clean_platform or not clean_chat_id:
        return False
    exact = {clean_chat_id, f"{clean_platform}:{clean_chat_id}"}
    wildcard = f"{clean_platform}:*"
    for entry in allowed:
        if not isinstance(entry, str):
            continue
        item = entry.strip()
        if not item:
            continue
        if item in exact or item == wildcard:
            return True
    return False


def _import_contextops(repo_path: str):
    """Lazily import the M4/M5 modules, optionally via a temporary sys.path
    insertion of the configured ContextOps repo (never persisted)."""
    inserted = False
    if repo_path:
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            inserted = True
    try:
        m4_mod = importlib.import_module(_M4_MODULE)
        m5_mod = importlib.import_module(_M5_MODULE)
        return m4_mod, m5_mod
    finally:
        if inserted:
            try:
                sys.path.remove(repo_path)
            except ValueError:
                pass


def _block_text_is_safe(text: str, cfg: dict) -> bool:
    for needle in _FORBIDDEN_SUBSTRINGS:
        if needle in text:
            return False
    if _PATH_SHAPE_RE.search(text):
        return False
    # Never echo configured secrets, hashes, or artifact paths.
    for key in ("approval_token", "expected_sidecar_hash",
                "m3_preview_path", "m3_receipt_path", "repo_path"):
        value = cfg.get(key)
        if isinstance(value, str) and value and value in text:
            return False
    if "tmp" in text.lower():
        return False
    return True


def _safe_semantic_line(value: object, cfg: dict) -> str | None:
    """Return one safe semantic ContextPack line, or ``None``.

    The M3 preview carries ``*_redacted`` strings intended for safe operator
    surfaces. The live gateway hook still treats them as untrusted: trim, cap
    length, require a single line, reject configured secret/path/hash values and
    provider/path-shaped output, then let the final whole-block scan re-check
    the assembled text.
    """

    if not isinstance(value, str):
        return None
    line = " ".join(value.strip().split())
    if not line:
        return None
    if len(line) > _MAX_SEMANTIC_CHARS:
        line = line[: _MAX_SEMANTIC_CHARS - 1].rstrip() + "…"
    if not _block_text_is_safe(line, cfg):
        return None
    return line


def _semantic_items(cfg: dict) -> tuple[list[str], list[str]] | None:
    if cfg.get("detail_level") != "semantic":
        return None
    try:
        preview_path = Path(str(cfg.get("m3_preview_path") or ""))
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    restore_raw = payload.get("restore_redacted")
    avoid_raw = payload.get("avoid_redacted")
    if not isinstance(restore_raw, list) or not isinstance(avoid_raw, list):
        return None
    restore: list[str] = []
    avoid: list[str] = []
    for raw in restore_raw[:_MAX_SEMANTIC_ITEMS]:
        line = _safe_semantic_line(raw, cfg)
        if line is None:
            return None
        restore.append(line)
    for raw in avoid_raw[:_MAX_SEMANTIC_ITEMS]:
        line = _safe_semantic_line(raw, cfg)
        if line is None:
            return None
        avoid.append(line)
    return restore, avoid


def _render_block(result, cfg: dict) -> str | None:
    lane = getattr(result, "lane", None)
    if not isinstance(lane, str) or not _SAFE_ID_RE.match(lane):
        return None
    restore_count = getattr(result, "restore_count", 0)
    avoid_count = getattr(result, "avoid_count", 0)
    if not isinstance(restore_count, int) or not isinstance(avoid_count, int):
        return None
    if not (0 <= restore_count <= 1000 and 0 <= avoid_count <= 1000):
        return None

    lines = [
        "[ContextOps one-turn hydration canary — metadata-only]",
        f"lane: {lane}",
        f"restore items: {restore_count} | avoid items: {avoid_count}",
    ]
    semantic = _semantic_items(cfg)
    if semantic is not None:
        restore_items, avoid_items = semantic
        lines.append("restore:")
        for item in restore_items:
            lines.append(f"- {item}")
        lines.append("avoid:")
        for item in avoid_items:
            lines.append(f"- {item}")
    else:
        for i in range(1, min(restore_count, _MAX_PLACEHOLDER_LINES) + 1):
            lines.append(f"ContextPack restore item #{i} (redacted/metadata-only)")
    lines.append(
        "note: preview-only metadata; no durable memory write, no live side effects"
    )
    block = "\n".join(lines)
    if not _block_text_is_safe(block, cfg):
        return None
    return block


def build_contextops_injection(config, *, platform: str, chat_id: str,
                               message_text: str = "") -> str | None:
    """Return a safe one-turn ContextOps block, or ``None`` (no injection).

    Every failure mode — disabled, wrong channel, missing config, import
    error, M4/M5 BLOCK, unexpected side-effect flags, leaky output, or any
    exception — fails closed to ``None``. A ``None`` return must never be
    surfaced as an error in model context.
    """
    try:
        cfg = _hydration_config(config)
        if cfg is None:
            return None
        if not _channel_allowed(cfg, platform=str(platform or ""),
                                chat_id=str(chat_id or "")):
            return None

        fields = {}
        for key in _REQUIRED_FIELDS:
            value = cfg.get(key)
            if not isinstance(value, str) or not value.strip():
                return None
            fields[key] = value

        m4_mod, m5_mod = _import_contextops(str(cfg.get("repo_path") or ""))

        fixture_text = message_text if isinstance(message_text, str) and message_text else _FIXTURE_MESSAGE

        with tempfile.TemporaryDirectory(prefix="hermes-contextops-") as td:
            message_path = Path(td) / "one_turn_message.txt"
            message_path.write_text(fixture_text, encoding="utf-8")

            assembly = m4_mod.runtime_assembly_preview_m4(
                fields["m3_preview_path"],
                fields["m3_receipt_path"],
                lane=fields["lane"],
                expected_sidecar_hash=fields["expected_sidecar_hash"],
                expect_effect_id=fields["expected_effect_id"],
                approval_token=fields["approval_token"],
                message_path=message_path,
                enabled=True,
            )
            if getattr(assembly, "verdict", "") != "GO":
                return None

            assembly_path = Path(td) / "assembly.json"
            assembly_path.write_text(
                json.dumps(assembly.model_dump(mode="json")), encoding="utf-8")

            result = m5_mod.one_turn_injection_preview_m5(
                assembly_path,
                lane=fields["lane"],
                target_id=fields["target_id"],
                effect_id=fields["expected_effect_id"],
                expected_sidecar_hash=fields["expected_sidecar_hash"],
                expected_assembly_hash=m5_mod.assembly_pin(assembly),
                approval_token=fields["approval_token"],
                approver=fields["approver"],
                reason=fields["reason"],
            )
            if getattr(result, "verdict", "") != "GO":
                return None
            if getattr(result, "injected", True) is not False:
                return None
            if getattr(result, "model_call", True) is not False:
                return None
            side_effects = getattr(result, "side_effects", None)
            if isinstance(side_effects, dict) and any(side_effects.values()):
                return None

            return _render_block(result, cfg)
    except Exception:
        return None


def maybe_append_contextops_context(context_prompt: str, config, *,
                                    platform: str, chat_id: str,
                                    message_text: str = "") -> str:
    """Append the canary block to ``context_prompt`` iff the gates pass."""
    block = build_contextops_injection(
        config, platform=platform, chat_id=chat_id, message_text=message_text)
    if block:
        return f"{context_prompt}\n\n{block}"
    return context_prompt
