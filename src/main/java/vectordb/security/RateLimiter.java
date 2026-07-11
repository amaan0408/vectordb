package vectordb.security;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simple in-memory per-user rate limiter.
 * Resets every 60 seconds.
 */
@Component
public class RateLimiter {

    private final Map<String, AtomicInteger> requestCounts = new ConcurrentHashMap<>();
    private static final int MAX_REQUESTS_PER_MINUTE = 30;

    public boolean allow(String username) {
        AtomicInteger count = requestCounts.computeIfAbsent(username, u -> new AtomicInteger(0));
        return count.incrementAndGet() <= MAX_REQUESTS_PER_MINUTE;
    }

    @Scheduled(fixedRate = 60000)
    public void resetCounts() {
        requestCounts.clear();
    }
}