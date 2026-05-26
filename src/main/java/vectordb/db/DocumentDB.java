package vectordb.db;

import vectordb.distance.DistanceMetrics;
import vectordb.index.BruteForce;
import vectordb.index.HNSW;
import vectordb.model.DocItem;
import vectordb.model.VectorItem;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * HNSW-only index for real Ollama embeddings (768D).
 * Equivalent to C++ class DocumentDB.
 */
public class DocumentDB {

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce(); // fallback for small sets
    private final AtomicInteger nextId = new AtomicInteger(1);
    private int dims = 0;

    /** Insert one chunk with its pre-computed embedding. */
    public synchronized int insert(String title, String text, float[] emb) {
        if (dims == 0) dims = emb.length;
        DocItem item = new DocItem(nextId.getAndIncrement(), title, text, emb);
        store.put(item.id, item);
        VectorItem vi = new VectorItem(item.id, title, "doc", emb);
        hnsw.insert(vi, DistanceMetrics::cosine);
        bf.insert(vi);
        return item.id;
    }

    /**
     * Semantic search — returns top-k most similar chunks with (distance, item) pairs.
     * Equivalent to C++ DocumentDB::search().
     */
    public synchronized List<float[]> searchRaw(float[] q, int k) {
        if (store.isEmpty()) return Collections.emptyList();
        List<float[]> raw = (store.size() < 10)
                ? bf.knn(q, k, DistanceMetrics::cosine)
                : hnsw.knn(q, k, 50, DistanceMetrics::cosine);
        // Filter by max_dist = 0.7 (same as C++)
        List<float[]> out = new ArrayList<>();
        for (float[] pair : raw) {
            if (pair[0] <= 0.7f) out.add(pair);
        }
        return out;
    }

    public synchronized List<DocItemWithDist> search(float[] q, int k) {
        List<float[]> raw = searchRaw(q, k);
        List<DocItemWithDist> result = new ArrayList<>();
        for (float[] pair : raw) {
            int id = (int) pair[1];
            DocItem di = store.get(id);
            if (di != null) result.add(new DocItemWithDist(pair[0], di));
        }
        return result;
    }

    public static class DocItemWithDist {
        public float dist;
        public DocItem item;
        DocItemWithDist(float dist, DocItem item) { this.dist = dist; this.item = item; }
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        hnsw.remove(id);
        bf.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() { return new ArrayList<>(store.values()); }

    public synchronized int size() { return store.size(); }

    public synchronized int getDims() { return dims; }
}