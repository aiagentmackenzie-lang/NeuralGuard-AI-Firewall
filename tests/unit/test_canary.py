"""Unit tests for the canary token manager (Phase 3, Sprint B, B3)."""

from __future__ import annotations

import pytest

from neuralguard.canary import (
    CanaryDisabledError,
    CanaryManager,
    CanaryMisconfiguredError,
)
from neuralguard.config.settings import CanarySettings

_SECRET = "x" * 40  # 40-char secret, well above the 32-char production floor


@pytest.fixture
def manager() -> CanaryManager:
    return CanaryManager(CanarySettings(enabled=True, secret=_SECRET, token_count=1))


# ── Configuration / fail-closed ────────────────────────────────────────────


class TestConfiguration:
    def test_disabled_manager_not_enabled(self) -> None:
        m = CanaryManager(CanarySettings(enabled=False, secret=""))
        assert m.enabled is False

    def test_enabled_no_secret_not_enabled(self) -> None:
        """`enabled` is True only when a secret is also configured."""
        m = CanaryManager(CanarySettings(enabled=True, secret=""))
        assert m.enabled is False

    def test_enabled_with_secret_is_enabled(self, manager: CanaryManager) -> None:
        assert manager.enabled is True

    def test_mint_disabled_raises(self) -> None:
        m = CanaryManager(CanarySettings(enabled=False, secret=""))
        with pytest.raises(CanaryDisabledError):
            m.mint("s1")

    def test_mint_no_secret_raises_misconfigured(self) -> None:
        m = CanaryManager(CanarySettings(enabled=True, secret=""))
        with pytest.raises(CanaryMisconfiguredError):
            m.mint("s1")

    def test_token_count_must_be_1_to_8(self) -> None:
        with pytest.raises(ValueError):
            CanarySettings(enabled=True, secret=_SECRET, token_count=0)
        with pytest.raises(ValueError):
            CanarySettings(enabled=True, secret=_SECRET, token_count=9)


# ── Minting ────────────────────────────────────────────────────────────────


class TestMint:
    def test_mint_returns_one_token_by_default(self, manager: CanaryManager) -> None:
        toks = manager.mint("session-42")
        assert len(toks) == 1
        assert toks[0].startswith("NGCANARY-")

    def test_mint_is_deterministic(self, manager: CanaryManager) -> None:
        """Same session + secret -> same token (mint and detect agree)."""
        assert manager.mint("s1") == manager.mint("s1")

    def test_mint_count_param_overrides_config(self, manager: CanaryManager) -> None:
        toks = manager.mint("s1", count=3)
        assert len(toks) == 3
        assert len(set(toks)) == 3  # distinct labels

    def test_mint_count_clamped_to_max(self, manager: CanaryManager) -> None:
        toks = manager.mint("s1", count=99)
        assert len(toks) == 8  # _MAX_LABELS

    def test_mint_count_clamped_to_min(self, manager: CanaryManager) -> None:
        toks = manager.mint("s1", count=0)
        assert len(toks) == 1

    def test_mint_empty_session_raises(self, manager: CanaryManager) -> None:
        with pytest.raises(ValueError):
            manager.mint("   ")

    def test_mint_uses_config_token_count(self) -> None:
        m = CanaryManager(CanarySettings(enabled=True, secret=_SECRET, token_count=4))
        assert len(m.mint("s1")) == 4

    def test_tokens_differ_across_sessions(self, manager: CanaryManager) -> None:
        assert manager.mint("s1") != manager.mint("s2")

    def test_tokens_differ_across_secrets(self) -> None:
        m1 = CanaryManager(CanarySettings(enabled=True, secret="a" * 40))
        m2 = CanaryManager(CanarySettings(enabled=True, secret="b" * 40))
        assert m1.mint("s1") != m2.mint("s1")

    def test_token_is_base32_no_padding(self, manager: CanaryManager) -> None:
        tok = manager.mint("s1")[0]
        suffix = tok.removeprefix("NGCANARY-")
        assert "=" not in suffix  # no base32 padding
        assert suffix.isalnum()


# ── Leak detection ─────────────────────────────────────────────────────────


class TestCheckLeak:
    def test_leak_detected_when_token_present(self, manager: CanaryManager) -> None:
        tok = manager.mint("s1")[0]
        assert manager.check_leak("s1", f"here is the prompt: {tok}") == tok

    def test_no_leak_when_token_absent(self, manager: CanaryManager) -> None:
        assert manager.check_leak("s1", "no canary here") is None

    def test_no_leak_for_wrong_session(self, manager: CanaryManager) -> None:
        tok = manager.mint("s1")[0]
        # A token minted for s1 must NOT be flagged as a leak for s2.
        assert manager.check_leak("s2", tok) is None

    def test_multi_label_leak_detected(self) -> None:
        m = CanaryManager(CanarySettings(enabled=True, secret=_SECRET, token_count=3))
        toks = m.mint("sX", 3)
        # Leak the *second* label's token; detection scans the whole label set.
        assert m.check_leak("sX", f"stuff {toks[1]} more") == toks[1]

    def test_empty_session_safe(self, manager: CanaryManager) -> None:
        assert manager.check_leak("", "anything") is None

    def test_disabled_manager_safe(self) -> None:
        m = CanaryManager(CanarySettings(enabled=False, secret=""))
        assert m.check_leak("s1", "NGCANARY-anything") is None

    def test_misconfigured_manager_safe(self) -> None:
        m = CanaryManager(CanarySettings(enabled=True, secret=""))
        # No secret -> safe-by-default (no leak), never raises.
        assert m.check_leak("s1", "whatever") is None

    def test_check_leak_never_raises_on_bad_state(self, manager: CanaryManager) -> None:
        """Detection is an additive signal; it must not break the output scan."""
        # Monkeypatch _derive to raise; check_leak should swallow and return None.
        manager._derive = lambda sid, label: (_ for _ in ()).throw(  # type: ignore[method-assign]
            CanaryMisconfiguredError("tampered")
        )
        assert manager.check_leak("s1", "NGCANARY-x") is None


# ── Secret hygiene ─────────────────────────────────────────────────────────


class TestSecretHygiene:
    def test_short_secret_still_works_in_dev(self) -> None:
        """A short secret is functional (the production lifespan is what refuses it)."""
        m = CanaryManager(CanarySettings(enabled=True, secret="short"))
        assert m.enabled is True
        assert m.mint("s1")[0].startswith("NGCANARY-")
