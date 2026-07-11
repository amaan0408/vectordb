package security;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for per-user rate limiting logic.
 */
class RateLimiterTest {

    private RateLimiter rateLimiter;

    @BeforeEach
    void setUp() {
        rateLimiter = new RateLimiter();
    }

    @Test
    void allow_firstRequest_returnsTrue() {
        assertTrue(rateLimiter.allow("amaan"));
    }

    @Test
    void allow_withinLimit_alwaysReturnsTrue() {
        for (int i = 0; i < 30; i++) {
            assertTrue(rateLimiter.allow("amaan"), "Request " + (i+1) + " should be allowed");
        }
    }

    @Test
    void allow_exceedLimit_returnsFalse() {
        for (int i = 0; i < 30; i++) rateLimiter.allow("amaan");
        assertFalse(rateLimiter.allow("amaan"), "31st request should be denied");
    }

    @Test
    void allow_differentUsers_independentCounters() {
        for (int i = 0; i < 30; i++) rateLimiter.allow("amaan");

        // amaan is rate limited but testuser should still pass
        assertFalse(rateLimiter.allow("amaan"));
        assertTrue(rateLimiter.allow("testuser"), "Different user should not be affected");
    }

    @Test
    void resetCounts_allowsRequestsAgain() {
        for (int i = 0; i < 31; i++) rateLimiter.allow("amaan");
        assertFalse(rateLimiter.allow("amaan"));

        rateLimiter.resetCounts();

        assertTrue(rateLimiter.allow("amaan"), "After reset, requests should be allowed again");
    }
}
