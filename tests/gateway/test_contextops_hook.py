"""Tests for the ContextOps gateway hydration canary hook.

The hook (``gateway/contextops_hook.py``) assembles a one-turn, metadata-only
ContextOps injection block through the reviewed M4/M5 preview contracts. It is
default-off, scoped to a single allowlisted channel, and fail-closed on any
missing/unsafe artifact, import failure, or leaky output.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from gateway.contextops_hook import (
    build_contextops_injection,
    maybe_append_contextops_context,
)

CONTEXTOPS_REPO = Path("/home/duckran/dev/contextops-ese")

M4_MODULE = "contextops.integration.runtime_assembly_preview"
M5_MODULE = "contextops.integration.one_turn_injection_gate"

CHANNEL = "discord:1497895797579190357"
PLATFORM = "discord"
CHAT_ID = "1497895797579190357"

APPROVAL_TOKEN = "tok-canary-secret-0123456789"
SIDECAR_HASH = "sidecarhash0123456789abcdef"


def enabled_config(channel: str = CHANNEL, **overrides) -> dict:
    cfg = {
        "enabled": True,
        "allowed_channels": [channel],
        "repo_path": "",
        "lane": "canary-lane",
        "m3_preview_path": "/tmp/contextops/m3_preview.json",
        "m3_receipt_path": "/tmp/contextops/m3_receipt.json",
        "expected_sidecar_hash": SIDECAR_HASH,
        "expected_effect_id": "effect-001",
        "approval_token": APPROVAL_TOKEN,
        "approver": "duckran",
        "reason": "answer-path hydration canary",
        "target_id": CHANNEL,
    }
    cfg.update(overrides)
    return {"contextops": {"gateway_hydration": cfg}}


class FakeAssembly:
    def __init__(self, verdict="GO", lane="canary-lane", restore_count=2, avoid_count=1):
        self.verdict = verdict
        self.lane = lane
        self.restore_count = restore_count
        self.avoid_count = avoid_count

    def model_dump(self, mode="python"):
        return {
            "verdict": self.verdict,
            "lane": self.lane,
            "restore_count": self.restore_count,
            "avoid_count": self.avoid_count,
        }


class FakeResult:
    def __init__(self, verdict="GO", lane="canary-lane", restore_count=2, avoid_count=1,
                 injected=False, model_call=False, side_effects=None):
        self.verdict = verdict
        self.lane = lane
        self.restore_count = restore_count
        self.avoid_count = avoid_count
        self.injected = injected
        self.model_call = model_call
        self.side_effects = side_effects if side_effects is not None else {
            "live_injection": False, "model_call": False, "dispatch": False,
        }
        self.block_reason = None


def install_fake_contextops(monkeypatch, *, assembly=None, result=None,
                            m4_exc=None, m5_exc=None):
    """Insert fake M4/M5 modules into sys.modules; return the call log."""
    calls = {"m4": [], "m5": [], "pin": []}
    assembly = assembly if assembly is not None else FakeAssembly()
    result = result if result is not None else FakeResult()

    m4_mod = types.ModuleType(M4_MODULE)

    def runtime_assembly_preview_m4(preview_path, receipt_path, **kwargs):
        calls["m4"].append({"preview_path": preview_path,
                            "receipt_path": receipt_path, **kwargs})
        if m4_exc is not None:
            raise m4_exc
        # The one-turn message fixture must exist at call time.
        assert kwargs.get("message_path") is not None
        assert Path(kwargs["message_path"]).exists()
        return assembly

    m4_mod.runtime_assembly_preview_m4 = runtime_assembly_preview_m4

    m5_mod = types.ModuleType(M5_MODULE)

    def assembly_pin(asm):
        calls["pin"].append(asm)
        return "asm-fakepin0123456789"

    def one_turn_injection_preview_m5(assembly_path, **kwargs):
        calls["m5"].append({"assembly_path": assembly_path, **kwargs})
        if m5_exc is not None:
            raise m5_exc
        assert Path(assembly_path).exists()
        return result

    m5_mod.assembly_pin = assembly_pin
    m5_mod.one_turn_injection_preview_m5 = one_turn_injection_preview_m5

    monkeypatch.setitem(sys.modules, M4_MODULE, m4_mod)
    monkeypatch.setitem(sys.modules, M5_MODULE, m5_mod)
    return calls


# ---------------------------------------------------------------------------
# 1. Default-off no-op
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_default_config_section_is_inert(self):
        from hermes_cli.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG["contextops"]["gateway_hydration"]
        assert cfg["enabled"] is False
        assert cfg["allowed_channels"] == []

    def test_default_config_no_injection(self, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG

        calls = install_fake_contextops(monkeypatch)
        block = build_contextops_injection(
            DEFAULT_CONFIG, platform=PLATFORM, chat_id=CHAT_ID)
        assert block is None
        assert calls["m4"] == [] and calls["m5"] == []

    def test_empty_config_no_injection(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        assert build_contextops_injection({}, platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []

    @pytest.mark.parametrize("flag", ["true", "True", 1, "yes", "on", None])
    def test_enabled_requires_literal_true(self, monkeypatch, flag):
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config(enabled=flag)
        assert build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []


# ---------------------------------------------------------------------------
# 2. Enabled but non-allowlisted channel no-op
# ---------------------------------------------------------------------------

class TestChannelGate:
    def test_non_allowlisted_channel_no_op(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config()
        assert build_contextops_injection(
            config, platform=PLATFORM, chat_id="9999999999") is None
        assert calls["m4"] == [] and calls["m5"] == []

    def test_empty_allowlist_no_op(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config(allowed_channels=[])
        assert build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []

    def test_multi_channel_allowlist_fails_closed(self, monkeypatch):
        """Canary scope is one channel: more than one entry fails closed."""
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config(allowed_channels=[CHANNEL, "discord:222"])
        assert build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []

    def test_allowlist_must_be_list(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config(allowed_channels=CHANNEL)
        assert build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []


# ---------------------------------------------------------------------------
# 3. Allowlisted + enabled path drives M4/M5 and returns a safe block
# ---------------------------------------------------------------------------

class TestEnabledPath:
    def test_allowlisted_channel_calls_m4_m5_and_returns_block(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        config = enabled_config()
        block = build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID)

        assert block is not None
        assert len(calls["m4"]) == 1
        assert len(calls["m5"]) == 1

        m4 = calls["m4"][0]
        assert str(m4["preview_path"]) == "/tmp/contextops/m3_preview.json"
        assert str(m4["receipt_path"]) == "/tmp/contextops/m3_receipt.json"
        assert m4["lane"] == "canary-lane"
        assert m4["expected_sidecar_hash"] == SIDECAR_HASH
        assert m4["expect_effect_id"] == "effect-001"
        assert m4["approval_token"] == APPROVAL_TOKEN
        assert m4["enabled"] is True

        m5 = calls["m5"][0]
        assert m5["lane"] == "canary-lane"
        assert m5["target_id"] == CHANNEL
        assert m5["effect_id"] == "effect-001"
        assert m5["expected_sidecar_hash"] == SIDECAR_HASH
        assert m5["expected_assembly_hash"] == "asm-fakepin0123456789"
        assert m5["approval_token"] == APPROVAL_TOKEN
        assert m5["approver"] == "duckran"
        assert m5["reason"] == "answer-path hydration canary"

        # Metadata-only placeholders, never ContextPack content.
        assert "ContextPack restore item #1 (redacted/metadata-only)" in block
        assert "ContextPack restore item #2 (redacted/metadata-only)" in block
        assert "canary-lane" in block

    def test_assembly_json_written_for_m5(self, monkeypatch):
        seen = {}
        calls = install_fake_contextops(monkeypatch)
        orig_m5 = sys.modules[M5_MODULE].one_turn_injection_preview_m5

        def spy(assembly_path, **kwargs):
            seen["doc"] = json.loads(Path(assembly_path).read_text(encoding="utf-8"))
            return orig_m5(assembly_path, **kwargs)

        sys.modules[M5_MODULE].one_turn_injection_preview_m5 = spy
        block = build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID)
        assert block is not None
        assert seen["doc"]["verdict"] == "GO"
        assert len(calls["pin"]) == 1

    def test_live_message_text_is_supplied_to_m4_fixture_not_rendered(self, monkeypatch):
        install_fake_contextops(monkeypatch)
        m4_mod = sys.modules[M4_MODULE]
        orig_m4 = m4_mod.runtime_assembly_preview_m4
        seen = {}

        def spy(preview_path, receipt_path, **kwargs):
            seen["message"] = Path(kwargs["message_path"]).read_text(encoding="utf-8")
            return orig_m4(preview_path, receipt_path, **kwargs)

        setattr(m4_mod, "runtime_assembly_preview_m4", spy)
        raw_turn = "operator asks about the current ContextOps canary"
        block = build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID,
            message_text=raw_turn,
        )
        assert seen["message"] == raw_turn
        assert block is not None
        assert raw_turn not in block

    def test_semantic_detail_renders_safe_m3_redacted_labels(self, monkeypatch, tmp_path):
        install_fake_contextops(monkeypatch)
        preview = tmp_path / "m3_preview.json"
        preview.write_text(json.dumps({
            "restore_redacted": [
                "Restore stance: keep active ContextOps thread and unresolved tension in view.",
                "Restore unresolved tension: active hydration must remain metadata-only.",
            ],
            "avoid_redacted": [
                "Do not treat compaction as a summary; preserve unresolved core.",
            ],
        }), encoding="utf-8")

        block = build_contextops_injection(
            enabled_config(m3_preview_path=str(preview), detail_level="semantic"),
            platform=PLATFORM,
            chat_id=CHAT_ID,
        )

        assert block is not None
        assert "restore:" in block
        assert "- Restore stance: keep active ContextOps thread" in block
        assert "avoid:" in block
        assert "- Do not treat compaction as a summary" in block
        assert "ContextPack restore item #1" not in block
        assert str(preview) not in block

    def test_semantic_detail_unsafe_redacted_label_falls_back_to_counts(self, monkeypatch, tmp_path):
        install_fake_contextops(monkeypatch)
        preview = tmp_path / "m3_preview.json"
        preview.write_text(json.dumps({
            "restore_redacted": ["Restore unsafe /home/duckran/private/path"],
            "avoid_redacted": ["Do not leak secrets"],
        }), encoding="utf-8")

        block = build_contextops_injection(
            enabled_config(m3_preview_path=str(preview), detail_level="semantic"),
            platform=PLATFORM,
            chat_id=CHAT_ID,
        )

        assert block is not None
        assert "ContextPack restore item #1 (redacted/metadata-only)" in block
        assert "/home/duckran" not in block


# ---------------------------------------------------------------------------
# 4. Unsafe/leaky ContextOps output fails closed
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_m4_block_verdict_no_injection(self, monkeypatch):
        calls = install_fake_contextops(
            monkeypatch, assembly=FakeAssembly(verdict="BLOCK"))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m5"] == []

    def test_m5_block_verdict_no_injection(self, monkeypatch):
        install_fake_contextops(monkeypatch, result=FakeResult(verdict="BLOCK"))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_leaky_lane_fails_closed(self, monkeypatch):
        install_fake_contextops(
            monkeypatch,
            result=FakeResult(lane="/home/duckran/dev/contextops-ese"))
        assert build_contextops_injection(
            enabled_config(lane="/home/duckran/dev/contextops-ese"),
            platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_side_effect_flag_fails_closed(self, monkeypatch):
        install_fake_contextops(
            monkeypatch,
            result=FakeResult(side_effects={"live_injection": True}))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_injected_true_fails_closed(self, monkeypatch):
        install_fake_contextops(monkeypatch, result=FakeResult(injected=True))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_m4_exception_fails_closed(self, monkeypatch):
        install_fake_contextops(monkeypatch, m4_exc=RuntimeError("boom"))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_m5_exception_fails_closed(self, monkeypatch):
        install_fake_contextops(monkeypatch, m5_exc=RuntimeError("boom"))
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_import_failure_fails_closed(self, monkeypatch):
        # ``None`` in sys.modules makes importlib raise ImportError.
        monkeypatch.setitem(sys.modules, M4_MODULE, None)
        monkeypatch.setitem(sys.modules, M5_MODULE, None)
        assert build_contextops_injection(
            enabled_config(), platform=PLATFORM, chat_id=CHAT_ID) is None

    def test_missing_required_field_fails_closed(self, monkeypatch):
        calls = install_fake_contextops(monkeypatch)
        assert build_contextops_injection(
            enabled_config(approval_token=""),
            platform=PLATFORM, chat_id=CHAT_ID) is None
        assert calls["m4"] == []


# ---------------------------------------------------------------------------
# 5. No raw transcript/path/token/provider-JSON leakage in the block
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_block_has_no_paths_tokens_or_json(self, monkeypatch):
        install_fake_contextops(monkeypatch)
        config = enabled_config()
        block = build_contextops_injection(
            config, platform=PLATFORM, chat_id=CHAT_ID)
        assert block is not None

        # No path fragments (the lone "/" in the "redacted/metadata-only"
        # placeholder is fine), no JSON braces, no secrets.
        assert not re.search(r"(?:^|[\s:=\"'(\[])/", block)  # token starting with /
        assert "//" not in block
        assert not re.search(r"/[A-Za-z0-9_.-]+/", block)  # multi-segment path
        assert "/home" not in block and "duckran" not in block
        assert "\\" not in block
        assert "{" not in block and "}" not in block
        assert APPROVAL_TOKEN not in block
        assert SIDECAR_HASH not in block
        assert "m3_preview" not in block and "m3_receipt" not in block
        assert "tmp" not in block.lower()
        assert not re.search(r"sk-[A-Za-z0-9]", block)


# ---------------------------------------------------------------------------
# 6. ContextOps core stays Hermes-free
# ---------------------------------------------------------------------------

class TestContextOpsCoreBoundary:
    def test_contextops_core_has_no_hermes_imports(self):
        if not CONTEXTOPS_REPO.exists():
            pytest.skip("ContextOps repo not present in this environment")
        # Match imports whose top-level root is a Hermes package. ContextOps'
        # own ``contextops.integration.hermes_adapter`` (root ``contextops``)
        # is a data-format adapter inside ContextOps, not a Hermes import.
        hermes_roots = r"(?:hermes|hermes_cli|hermes_agent|gateway|tui_gateway|acp_adapter)"
        pattern = re.compile(
            rf"^\s*(?:import\s+{hermes_roots}(?:[.\s,]|$)|from\s+{hermes_roots}(?:\.\S*)?\s+import)",
            re.MULTILINE,
        )
        offenders = []
        for path in (CONTEXTOPS_REPO / "contextops").rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(CONTEXTOPS_REPO)))
        assert offenders == []


# ---------------------------------------------------------------------------
# 7. Gateway integration appends the block only for allowed channel/config
# ---------------------------------------------------------------------------

class TestGatewayIntegration:
    def test_append_helper_appends_only_when_allowed(self, monkeypatch):
        install_fake_contextops(monkeypatch)
        base = "existing session context"
        config = enabled_config()

        out = maybe_append_contextops_context(
            base, config, platform=PLATFORM, chat_id=CHAT_ID)
        assert out.startswith(base + "\n\n")
        assert "ContextPack restore item #1 (redacted/metadata-only)" in out

        # Wrong channel: untouched.
        assert maybe_append_contextops_context(
            base, config, platform=PLATFORM, chat_id="other") == base
        # Disabled: untouched.
        assert maybe_append_contextops_context(
            base, enabled_config(enabled=False),
            platform=PLATFORM, chat_id=CHAT_ID) == base

    def test_run_py_wires_hook_on_answer_path(self):
        src = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
        text = src.read_text(encoding="utf-8")
        idx_ctx = text.index("context_prompt = build_session_context_prompt(")
        idx_prepare = text.index("message_text = await self._prepare_inbound_message_text(")
        idx_hook = text.index("maybe_append_contextops_context")
        idx_run = text.index("agent_result = await self._run_agent(")
        assert idx_ctx < idx_prepare < idx_hook < idx_run
        assert "message_text=message_text" in text[idx_hook:idx_run]
