package index;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import vectordb.distance.DistanceMetrics;
import vectordb.index.KDTree;
import vectordb.model.VectorItem;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class KDTreeTest {

    private KDTree kdt;

    @BeforeEach
    void setUp() {
        kdt = new KDTree(4);
        kdt.insert(new VectorItem(1, "A", "cs",   new float[]{0.9f, 0.1f, 0.1f, 0.1f}));
        kdt.insert(new VectorItem(2, "B", "math", new float[]{0.1f, 0.9f, 0.1f, 0.1f}));
        kdt.insert(new VectorItem(3, "C", "food", new float[]{0.1f, 0.1f, 0.9f, 0.1f}));
    }

    @Test
    void knn_findsExactNearest() {
        float[] q = {0.9f, 0.1f, 0.1f, 0.1f};
        List<float[]> r = kdt.knn(q, 1, DistanceMetrics::euclidean);
        assertEquals(1, (int) r.get(0)[1]);
    }

    @Test
    void knn_resultsSortedAscending() {
        float[] q = {0.5f, 0.5f, 0.1f, 0.1f};
        List<float[]> r = kdt.knn(q, 3, DistanceMetrics::euclidean);
        for (int i = 1; i < r.size(); i++) {
            assertTrue(r.get(i)[0] >= r.get(i-1)[0]);
        }
    }

    @Test
    void rebuild_afterRemoval_excludesDeletedItem() {
        // KDTree doesn't support direct delete — rebuild without item 1
        kdt.rebuild(List.of(
            new VectorItem(2, "B", "math", new float[]{0.1f, 0.9f, 0.1f, 0.1f}),
            new VectorItem(3, "C", "food", new float[]{0.1f, 0.1f, 0.9f, 0.1f})
        ));

        float[] q = {0.9f, 0.1f, 0.1f, 0.1f};
        List<float[]> r = kdt.knn(q, 3, DistanceMetrics::euclidean);
        assertTrue(r.stream().noneMatch(res -> (int)res[1] == 1));
    }

    @Test
    void knn_kLargerThanNodes_returnsAll() {
        float[] q = {0.5f, 0.5f, 0.5f, 0.5f};
        List<float[]> r = kdt.knn(q, 100, DistanceMetrics::euclidean);
        assertEquals(3, r.size());
    }
}
