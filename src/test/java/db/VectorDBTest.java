package vectordb.db;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import vectordb.distance.DistanceMetrics;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VectorDB — the unified interface over all three algorithms.
 */
class VectorDBTest {

    private VectorDB db;

    @BeforeEach
    void setUp() {
        db = new VectorDB(4);
        db.insert("CS item",    "cs",    new float[]{0.9f, 0.1f, 0.1f, 0.1f}, DistanceMetrics.get("cosine"));
        db.insert("Math item",  "math",  new float[]{0.1f, 0.9f, 0.1f, 0.1f}, DistanceMetrics.get("cosine"));
        db.insert("Food item",  "food",  new float[]{0.1f, 0.1f, 0.9f, 0.1f}, DistanceMetrics.get("cosine"));
        db.insert("Sport item", "sports",new float[]{0.1f, 0.1f, 0.1f, 0.9f}, DistanceMetrics.get("cosine"));
    }

    // ── Insert & Size ──────────────────────────────────────────────────

    @Test
    void insert_incrementsSize() {
        assertEquals(4, db.size());
        db.insert("New", "cs", new float[]{0.5f, 0.5f, 0.1f, 0.1f}, DistanceMetrics.get("cosine"));
        assertEquals(5, db.size());
    }

    @Test
    void insert_returnsIncrementingId() {
        int id = db.insert("X", "cs", new float[]{0.5f,0.5f,0.1f,0.1f}, DistanceMetrics.get("cosine"));
        assertEquals(5, id);
    }

    // ── Remove ────────────────────────────────────────────────────────

    @Test
    void remove_existingItem_returnsTrue() {
        assertTrue(db.remove(1));
    }

    @Test
    void remove_nonExistentItem_returnsFalse() {
        assertFalse(db.remove(999));
    }

    @Test
    void remove_decrementsSizeByOne() {
        db.remove(1);
        assertEquals(3, db.size());
    }

    // ── Search — HNSW ─────────────────────────────────────────────────

    @Test
    void search_hnsw_returnsTopK() {
        VectorDB.SearchOut out = db.search(new float[]{0.9f,0.1f,0.1f,0.1f}, 2, "cosine", "hnsw");
        assertEquals(2, out.hits.size());
    }

    @Test
    void search_hnsw_nearestHitFirst() {
        float[] q = {0.9f, 0.1f, 0.1f, 0.1f};
        VectorDB.SearchOut out = db.search(q, 1, "cosine", "hnsw");
        assertEquals("CS item", out.hits.get(0).meta);
    }

    // ── Search — BruteForce ───────────────────────────────────────────

    @Test
    void search_bruteforce_nearestHitFirst() {
        float[] q = {0.1f, 0.9f, 0.1f, 0.1f};
        VectorDB.SearchOut out = db.search(q, 1, "cosine", "bruteforce");
        assertEquals("Math item", out.hits.get(0).meta);
    }

    // ── Search — KDTree ───────────────────────────────────────────────

    @Test
    void search_kdtree_nearestHitFirst() {
        float[] q = {0.1f, 0.1f, 0.9f, 0.1f};
        VectorDB.SearchOut out = db.search(q, 1, "euclidean", "kdtree");
        assertEquals("Food item", out.hits.get(0).meta);
    }

    // ── Latency field is set ──────────────────────────────────────────

    @Test
    void search_latencyFieldIsPopulated() {
        VectorDB.SearchOut out = db.search(new float[]{0.5f,0.5f,0.1f,0.1f}, 2, "cosine", "hnsw");
        assertTrue(out.latencyUs >= 0);
    }

    // ── All items ─────────────────────────────────────────────────────

    @Test
    void all_returnsAllInsertedItems() {
        assertEquals(4, db.all().size());
    }

    // ── Benchmark ─────────────────────────────────────────────────────

    @Test
    void benchmark_allThreeLatenciesNonNegative() {
        VectorDB.BenchOut b = db.benchmark(new float[]{0.5f,0.5f,0.1f,0.1f}, 2, "cosine");
        assertTrue(b.bfUs   >= 0);
        assertTrue(b.kdUs   >= 0);
        assertTrue(b.hnswUs >= 0);
        assertEquals(4, b.n);
    }
}
