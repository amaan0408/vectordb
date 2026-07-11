package index;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import vectordb.distance.DistanceMetrics;
import vectordb.model.VectorItem;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for BruteForce exact nearest-neighbour search.
 */
class BruteForceTest {

    private BruteForce bf;

    @BeforeEach
    void setUp() {
        bf = new BruteForce();
        bf.insert(new VectorItem(1, "CS item",    "cs",    new float[]{0.9f, 0.1f, 0.1f, 0.1f}));
        bf.insert(new VectorItem(2, "Math item",  "math",  new float[]{0.1f, 0.9f, 0.1f, 0.1f}));
        bf.insert(new VectorItem(3, "Food item",  "food",  new float[]{0.1f, 0.1f, 0.9f, 0.1f}));
        bf.insert(new VectorItem(4, "Sport item", "sports",new float[]{0.1f, 0.1f, 0.1f, 0.9f}));
    }

    @Test
    void knn_returnsCorrectNearestNeighbour() {
        float[] q = {0.9f, 0.1f, 0.1f, 0.1f};   // closest to item 1
        List<float[]> results = bf.knn(q, 1, DistanceMetrics::euclidean);

        assertEquals(1, results.size());
        assertEquals(1, (int) results.get(0)[1]);
    }

    @Test
    void knn_resultsSortedByDistanceAscending() {
        float[] q = {0.5f, 0.5f, 0.1f, 0.1f};
        List<float[]> results = bf.knn(q, 4, DistanceMetrics::euclidean);

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i)[0] >= results.get(i-1)[0],
                "Results must be sorted ascending by distance");
        }
    }

    @Test
    void knn_kLargerThanSize_returnsAll() {
        float[] q = {0.5f, 0.5f, 0.5f, 0.5f};
        List<float[]> results = bf.knn(q, 100, DistanceMetrics::euclidean);
        assertEquals(4, results.size());
    }

    @Test
    void remove_deletesItem_notReturnedInSearch() {
        bf.remove(1);
        float[] q = {0.9f, 0.1f, 0.1f, 0.1f};
        List<float[]> results = bf.knn(q, 1, DistanceMetrics::euclidean);

        assertTrue(results.stream().noneMatch(r -> (int)r[1] == 1),
            "Deleted item should not appear in results");
    }

    @Test
    void size_reflectsInsertAndRemove() {
        assertEquals(4, bf.size());
        bf.remove(1);
        assertEquals(3, bf.size());
    }

    @Test
    void knn_emptyIndex_returnsEmptyList() {
        BruteForce empty = new BruteForce();
        List<float[]> results = empty.knn(new float[]{1f, 0f}, 5, DistanceMetrics::euclidean);
        assertTrue(results.isEmpty());
    }

    @Test
    void knn_cosineDistance_correctNearest() {
        float[] q = {0f, 1f, 0f, 0f};   // direction of item 2
        List<float[]> results = bf.knn(q, 1, DistanceMetrics::cosine);
        assertEquals(2, (int) results.get(0)[1]);
    }
}
