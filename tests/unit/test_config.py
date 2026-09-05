"""Unit tests for configuration settings."""

import pytest

from neuralguard.config.settings import (
    ActionSettings,
    AuditSettings,
    AuthSettings,
    NeuralGuardConfig,
    RateLimitSettings,
    ScannerSettings,
    ServerSettings,
    TenantSettings,
    load_config,
)


class TestServerSettings:
    """Tests for ServerSettings."""

    def test_defaults(self):
        s = ServerSettings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.workers == 1
        assert s.log_level == "INFO"

    def test_valid_port(self):
        s = ServerSettings(port=443)
        assert s.port == 443

    def test_invalid_port_low(self):
        with pytest.raises(ValueError):
            ServerSettings(port=0)

    def test_invalid_port_high(self):
        with pytest.raises(ValueError):
            ServerSettings(port=70000)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEURALGUARD_PORT", "9000")
        monkeypatch.setenv("NEURALGUARD_HOST", "127.0.0.1")
        s = ServerSettings()
        assert s.port == 9000
        assert s.host == "127.0.0.1"


class TestScannerSettings:
    """Tests for ScannerSettings."""

    def test_defaults(self):
        s = ScannerSettings()
        assert s.max_input_length == 32_000
        assert s.max_decompression_ratio == 10.0
        assert s.regex_timeout_ms == 50
        assert s.semantic_enabled is False
        assert s.judge_enabled is False

    def test_semantic_toggle(self):
        s = ScannerSettings(semantic_enabled=True)
        assert s.semantic_enabled is True


class TestActionSettings:
    """Tests for ActionSettings."""

    def test_defaults(self):
        s = ActionSettings()
        assert s.fail_closed is True
        assert s.default_action == "block"
        assert s.score_threshold_block == 0.85
        assert s.score_threshold_sanitize == 0.60


class TestNeuralGuardConfig:
    """Tests for top-level config."""

    def test_defaults(self):
        c = NeuralGuardConfig()
        assert c.app_name == "NeuralGuard"
        assert c.version == "0.2.0"
        assert c.environment == "development"
        assert isinstance(c.server, ServerSettings)
        assert isinstance(c.scanner, ScannerSettings)
        assert isinstance(c.action, ActionSettings)
        assert isinstance(c.audit, AuditSettings)
        assert isinstance(c.tenant, TenantSettings)
        assert isinstance(c.rate_limit, RateLimitSettings)

    def test_load_config(self):
        c = load_config()
        assert isinstance(c, NeuralGuardConfig)

    def test_nested_env_override(self, monkeypatch):
        monkeypatch.setenv("NEURALGUARD_SCANNER_MAX_INPUT_LENGTH", "50000")
        c = NeuralGuardConfig()
        assert c.scanner.max_input_length == 50000


class TestAuthSettingsEnvParsing:
    """Regression: the key|tenant format must parse from env.

    pydantic-settings 2.14 raises SettingsError for a list[str] field when
    the env value is not JSON. NoDecode + the parse_api_keys before-validator
    keeps the documented comma-separated 'key|tenant' format working.
    """

    def test_pipe_tenant_format_from_env(self, monkeypatch):
        monkeypatch.setenv("NEURALGUARD_AUTH_API_KEYS", "strong-key-1|acme,strong-key-2|demo")
        s = AuthSettings()
        assert s.key_to_tenant() == {"strong-key-1": "acme", "strong-key-2": "demo"}

    def test_bare_key_defaults_to_default_tenant(self, monkeypatch):
        monkeypatch.setenv("NEURALGUARD_AUTH_API_KEYS", "strong-key-1")
        s = AuthSettings()
        assert s.key_to_tenant() == {"strong-key-1": "default"}

    def test_in_code_list_still_works(self):
        s = AuthSettings(api_keys=["k1|a", "k2"])
        assert s.key_to_tenant() == {"k1": "a", "k2": "default"}


class TestRateLimitSettingsBackend:
    def test_defaults_to_memory(self):
        assert RateLimitSettings().backend == "memory"

    def test_redis_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("NEURALGUARD_RATELIMIT_BACKEND", "redis")
        monkeypatch.setenv("NEURALGUARD_RATELIMIT_REDIS_URL", "redis://localhost:6379/0")
        s = RateLimitSettings()
        assert s.backend == "redis"
        assert s.redis_url == "redis://localhost:6379/0"
