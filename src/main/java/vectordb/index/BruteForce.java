package vectordb.index;

import vectordb.distance.DistFn;
import vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Exact O(N·d) nearest-neighbour search — direct translation of C++ class BruteForce.
 */
public class BruteForce {

    private final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem v) {
        items.add(v);
    }

    public synchronized void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    /**
     * Returns up to k (distance, id) pairs sorted by ascending distance.
     * Uses float[2]: [0]=distance  [1]=id (cast to float for simplicity).
     */
    public synchronized List<float[]> knn(float[] q, int k, DistFn dist) {
        List<float[]> r = new ArrayList<>(items.size());
        for (VectorItem v : items) {
            r.add(new float[]{dist.compute(q, v.emb), v.id});
        }
        r.sort((a, b) -> Float.compare(a[0], b[0]));
        if (r.size() > k) r = r.subList(0, k);
        return new ArrayList<>(r);
    }

    public synchronized List<VectorItem> all() {
        return new ArrayList<>(items);
    }

    public synchronized int size() { return items.size(); }
}