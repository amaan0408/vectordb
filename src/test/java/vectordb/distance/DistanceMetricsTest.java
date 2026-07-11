package vectordb.distance;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for all three distance functions.
 * These are pure math — no Spring context needed.
 */
class DistanceMetricsTest {

    // ── Euclidean ──────────────────────────────────────────────────────

    @Test
    void euclidean_identicalVectors_returnsZero() {
        float[] a = {1f, 2f, 3f};
        float[] b = {1f, 2f, 3f};
        assertEquals(0f, DistanceMetrics.euclidean(a, b), 1e-6f);
    }

    @Test
    void euclidean_knownDistance_correct() {
        float[] a = {0f, 0f};
        float[] b = {3f, 4f};
        assertEquals(5f, DistanceMetrics.euclidean(a, b), 1e-5f);
    }

    @Test
    void euclidean_isSymmetric() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        assertEquals(DistanceMetrics.euclidean(a, b),
                     DistanceMetrics.euclidean(b, a), 1e-6f);
    }

    // ── Cosine ────────────────────────────────────────────────────────

    @Test
    void cosine_identicalVectors_returnsZero() {
        float[] a = {1f, 0f, 0f};
        assertEquals(0f, DistanceMetrics.cosine(a, a), 1e-6f);
    }

    @Test
    void cosine_oppositeVectors_returnsTwo() {
        float[] a = {1f, 0f};
        float[] b = {-1f, 0f};
        assertEquals(2f, DistanceMetrics.cosine(a, b), 1e-5f);
    }

    @Test
    void cosine_perpendicularVectors_returnsOne() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(1f, DistanceMetrics.cosine(a, b), 1e-5f);
    }

    @Test
    void cosine_zeroVector_doesNotThrow() {
        float[] a = {0f, 0f, 0f};
        float[] b = {1f, 2f, 3f};
        assertDoesNotThrow(() -> DistanceMetrics.cosine(a, b));
    }

    @Test
    void cosine_scaleInvariant() {
        float[] a  = {1f, 2f, 3f};
        float[] b  = {2f, 4f, 6f};   // b = 2×a, same direction
        assertEquals(0f, DistanceMetrics.cosine(a, b), 1e-5f);
    }

    // ── Manhattan ─────────────────────────────────────────────────────

    @Test
    void manhattan_identicalVectors_returnsZero() {
        float[] a = {1f, 2f, 3f};
        assertEquals(0f, DistanceMetrics.manhattan(a, a), 1e-6f);
    }

    @Test
    void manhattan_knownDistance_correct() {
        float[] a = {1f, 2f};
        float[] b = {4f, 6f};
        assertEquals(7f, DistanceMetrics.manhattan(a, b), 1e-5f);
    }

    // ── Factory ───────────────────────────────────────────────────────

    @Test
    void get_cosine_returnsCosineFn() {
        DistFn fn = DistanceMetrics.get("cosine");
        float[] a = {1f, 0f};
        assertEquals(0f, fn.compute(a, a), 1e-6f);
    }

    @Test
    void get_unknown_defaultsToEuclidean() {
        DistFn fn = DistanceMetrics.get("whatever");
        float[] a = {0f, 0f};
        float[] b = {3f, 4f};
        assertEquals(5f, fn.compute(a, b), 1e-5f);
    }
}
