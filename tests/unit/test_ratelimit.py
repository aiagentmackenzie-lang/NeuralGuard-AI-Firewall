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
