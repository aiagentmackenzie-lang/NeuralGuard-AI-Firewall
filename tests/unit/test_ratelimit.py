"""Tests for rate limiter — bug fix validation."""

import time

from neuralguard.config.settings import RateLimitSettings
from neuralguard.middleware.ratelimit import SlidingWindowCounter


class TestSlidingWindowCounter:
    """Tests for C-02 burst logic and M-06 tenant cleanup."""

    def test_burst_allows_over_limit(self):
        """C-02: burst should allow requests up to limit + burst."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Use limit=5, burst=3
        # Should allow 5 + 3 = 8 requests before blocking
        for i in range(8):
            allowed, _remaining, _retry_after = counter.check("tenant1", limit=5, burst=3)
            assert allowed, f"Request {i + 1} should be allowed (limit=5, burst=3)"

        # 9th request should be blocked
        allowed, _remaining, _retry_after = counter.check("tenant1", limit=5, burst=3)
        assert not allowed, "Request 9 should be blocked (beyond limit + burst)"

    def test_remaining_decreases(self):
        """Remaining should decrease with each request."""
        counter = SlidingWindowCounter(window_seconds=60)

        _, remaining1, _ = counter.check("tenant1", limit=5, burst=3)
        assert remaining1 == 7  # 8 - 1 = 7 remaining

        _, remaining2, _ = counter.check("tenant1", limit=5, burst=3)
        assert remaining2 == 6  # 8 - 2 = 6 remaining

    def test_separate_tenants(self):
        """Different tenants should have separate counters."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Exhaust tenant1
        for _ in range(8):
            counter.check("tenant1", limit=5, burst=3)

        # Tenant2 should still be allowed
        allowed, _, _ = counter.check("tenant2", limit=5, burst=3)
        assert allowed

    def test_m06_inactive_tenant_cleanup(self):
        """M-06: Inactive tenants should be cleaned up periodically."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Add a tenant
        counter.check("inactive_tenant", limit=5, burst=3)
        assert "inactive_tenant" in counter._counters

        # Simulate aging: manually set timestamps to be old
        old_time = time.time() - 120  # 2 minutes ago
        counter._counters["inactive_tenant"] = [old_time]

        # Force cleanup by setting last cleanup to long ago
        counter._last_cleanup = time.time() - 600  # 10 minutes ago
        counter.check("new_tenant", limit=5, burst=3)

        # inactive_tenant should have been cleaned up
        assert "inactive_tenant" not in counter._counters


class TestCostBasedCounter:
    """F7: cost-based mode — token-estimate charges against a budget."""

    def test_cost_charged_against_budget(self):
        counter = SlidingWindowCounter(window_seconds=60)
        ok1 = counter.check_cost("t", 225, 1000)
        assert ok1[0] and ok1[1] == 775 and ok1[2] == 0
        ok2 = counter.check_cost("t", 225, 1000)
        assert ok2[0] and ok2[1] == 550
        # 450 used + 600 cost > 1000 -> rejected with retry_after
        ok3 = counter.check_cost("t", 600, 1000)
        assert not ok3[0]
        assert ok3[1] == 550  # remaining budget unchanged
        assert ok3[2] >= 1

    def test_request_exceeding_whole_budget_rejected(self):
        counter = SlidingWindowCounter(window_seconds=60)
        allowed, remaining, retry = counter.check_cost("t", 200, 100)
        assert not allowed
        assert remaining == 100  # budget untouched by the rejected request
        assert retry >= 1

    def test_window_expiry_frees_budget(self):
        counter = SlidingWindowCounter(window_seconds=60)
        assert counter.check_cost("t", 900, 1000)[0]
        # age the entry out of the window
        counter._cost_entries["t"] = [(time.time() - 120, 900)]
        allowed, remaining, _ = counter.check_cost("t", 500, 1000)
        assert allowed
        assert remaining == 500

    def test_cost_mode_store_is_separate_from_request_mode(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.check("t", limit=5, burst=3)
        assert "t" not in counter._cost_entries
        counter.check_cost("t2", 10, 100)
        assert "t2" not in counter._counters

    def test_inactive_tenant_cleanup_covers_cost_store(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.check_cost("inactive_tenant", 10, 100)
        assert "inactive_tenant" in counter._cost_entries
        counter._cost_entries["inactive_tenant"] = [(time.time() - 120, 10)]
        counter._last_cleanup = time.time() - 600
        counter.check_cost("new_tenant", 10, 100)
        assert "inactive_tenant" not in counter._cost_entries


class TestCostBasedKnobWiring:
    """F20-class guard: probe the declared fields directly after construction."""

    def test_cost_settings_wired(self):
        settings = RateLimitSettings(cost_based=True, cost_units_per_minute=123)
        assert settings.cost_based is True
        assert settings.cost_units_per_minute == 123
        assert RateLimitSettings().cost_based is False
        assert RateLimitSettings().cost_units_per_minute == 100_000
