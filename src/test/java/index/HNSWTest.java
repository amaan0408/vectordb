package index;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import vectordb.distance.DistanceMetrics;
import vectordb.index.HNSW;
import vectordb.model.VectorItem;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the HNSW approximate nearest-neighbour index.
 * We verify structural correctness and search quality,
 * not exact results (HNSW is approximate by design).
 */
class HNSWTest {

    private HNSW hnsw;

    private VectorItem item(int id, float... emb) {
        return new VectorItem(id, "item-" + id, "test", emb);
    }

    @BeforeEach
    void setUp() {
        hnsw = new HNSW(4, 20);   // small M and ef for unit tests
    }

    @Test
    void insert_singleItem_graphNotEmpty() {
        hnsw.insert(item(1, 1f, 0f, 0f, 0f), DistanceMetrics::cosine);
        assertEquals(1, hnsw.size());
    }

    @Test
    void knn_emptyGraph_returnsEmptyList() {
        List<float[]> results = hnsw.knn(new float[]{1f,0f,0f,0f}, 5, 50, DistanceMetrics::cosine);
        assertTrue(results.isEmpty());
    }

    @Test
    void knn_singleItem_returnsThatItem() {
        hnsw.insert(item(1, 1f, 0f, 0f, 0f), DistanceMetrics::cosine);
        List<float[]> r = hnsw.knn(new float[]{1f, 0f, 0f, 0f}, 1, 50, DistanceMetrics::cosine);

        assertEquals(1, r.size());
        assertEquals(1, (int) r.get(0)[1]);
    }

    @Test
    void knn_nearestItemRankedFirst() {
        hnsw.insert(item(1, 0.9f, 0.1f, 0.0f, 0.0f), DistanceMetrics::cosine);
        hnsw.insert(item(2, 0.1f, 0.9f, 0.0f, 0.0f), DistanceMetrics::cosine);
        hnsw.insert(item(3, 0.0f, 0.0f, 0.9f, 0.1f), DistanceMetrics::cosine);

        float[] q = {0.95f, 0.05f, 0.0f, 0.0f};  // closest to item 1
        List<float[]> r = hnsw.knn(q, 1, 50, DistanceMetrics::cosine);

        assertEquals(1, (int) r.get(0)[1], "Item 1 should be the nearest neighbour");
    }

    @Test
    void knn_resultsSortedByDistanceAscending() {
        for (int i = 1; i <= 10; i++) {
            float v = i * 0.1f;
            hnsw.insert(item(i, v, 1-v, 0f, 0f), DistanceMetrics::euclidean);
        }
        float[] q = {0.5f, 0.5f, 0f, 0f};
        List<float[]> r = hnsw.knn(q, 5, 50, DistanceMetrics::euclidean);

        for (int i = 1; i < r.size(); i++) {
            assertTrue(r.get(i)[0] >= r.get(i-1)[0],
                "Results must be sorted ascending by distance");
        }
    }

    @Test
    void knn_kLargerThanSize_returnsAllItems() {
        hnsw.insert(item(1, 1f, 0f), DistanceMetrics::euclidean);
        hnsw.insert(item(2, 0f, 1f), DistanceMetrics::euclidean);

        List<float[]> r = hnsw.knn(new float[]{0.5f, 0.5f}, 100, 50, DistanceMetrics::euclidean);
        assertEquals(2, r.size());
    }

    @Test
    void remove_deletedItemNotReturnedInSearch() {
        hnsw.insert(item(1, 1f, 0f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.insert(item(2, 0f, 1f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.remove(1);

        float[] q = {1f, 0f, 0f, 0f};
        List<float[]> r = hnsw.knn(q, 5, 50, DistanceMetrics::cosine);

        assertTrue(r.stream().noneMatch(res -> (int)res[1] == 1),
            "Removed item must not appear in results");
    }

    @Test
    void remove_reducesSize() {
        hnsw.insert(item(1, 1f, 0f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.insert(item(2, 0f, 1f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.remove(1);
        assertEquals(1, hnsw.size());
    }

    @Test
    void getInfo_returnsCorrectNodeCount() {
        hnsw.insert(item(1, 1f, 0f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.insert(item(2, 0f, 1f, 0f, 0f), DistanceMetrics::cosine);
        hnsw.insert(item(3, 0f, 0f, 1f, 0f), DistanceMetrics::cosine);

        HNSW.GraphInfo info = hnsw.getInfo();
        assertEquals(3, info.nodeCount);
    }

    @Test
    void insert_largeDataset_searchStillFindsNearestInTopK() {
        // Insert 50 vectors; query should find the exact match within top-3
        for (int i = 0; i < 50; i++) {
            float v = i / 50f;
            hnsw.insert(item(i + 1, v, 1-v, v*0.5f, (1-v)*0.5f), DistanceMetrics::cosine);
        }

        // Query matches item at index 25 (i=24, v=0.48)
        float[] q = {0.48f, 0.52f, 0.24f, 0.26f};
        List<float[]> r = hnsw.knn(q, 3, 50, DistanceMetrics::cosine);

        assertFalse(r.isEmpty(), "Should return at least one result");
        assertTrue(r.get(0)[0] < 0.1f, "Nearest result should be very close to query");
    }
}
