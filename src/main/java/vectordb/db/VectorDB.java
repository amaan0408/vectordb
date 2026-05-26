package vectordb.db;

import vectordb.distance.DistFn;
import vectordb.distance.DistanceMetrics;
import vectordb.index.BruteForce;
import vectordb.index.HNSW;
import vectordb.index.KDTree;
import vectordb.model.VectorItem;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Unified interface over BruteForce + KDTree + HNSW for the 16-D demo vectors.
 * Equivalent to C++ class VectorDB.
 */
public class VectorDB {

    public final int dims;

    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw = new HNSW(16, 200);
    private final AtomicInteger nextId = new AtomicInteger(1);

    public VectorDB(int dims) {
        this.dims = dims;
        this.kdt  = new KDTree(dims);
    }

    // ── Insert ─────────────────────────────────────────────────────────────
    public synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
        VectorItem v = new VectorItem(nextId.getAndIncrement(), meta, cat, emb);
        store.put(v.id, v);
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id;
    }

    // ── Remove ─────────────────────────────────────────────────────────────
    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        kdt.rebuild(new ArrayList<>(store.values()));
        return true;
    }

    // ── Search ─────────────────────────────────────────────────────────────
    public static class Hit {
        public int id; public String meta; public String cat; public float[] emb; public float dist;
    }
    public static class SearchOut {
        public List<Hit> hits; public long latencyUs; public String algo; public String metric;
    }

    public synchronized SearchOut search(float[] q, int k, String metric, String algo) {
        DistFn dfn = DistanceMetrics.get(metric);
        long t0 = System.nanoTime();

        List<float[]> raw = switch (algo) {
            case "bruteforce" -> bf.knn(q, k, dfn);
            case "kdtree"     -> kdt.knn(q, k, dfn);
            default           -> hnsw.knn(q, k, 50, dfn);
        };

        long us = (System.nanoTime() - t0) / 1000;

        SearchOut out = new SearchOut();
        out.hits = new ArrayList<>();
        out.latencyUs = us;
        out.algo = algo;
        out.metric = metric;

        for (float[] pair : raw) {
            int id = (int) pair[1];
            VectorItem vi = store.get(id);
            if (vi != null) {
                Hit h = new Hit();
                h.id = id; h.meta = vi.metadata; h.cat = vi.category;
                h.emb = vi.emb; h.dist = pair[0];
                out.hits.add(h);
            }
        }
        return out;
    }

    // ── Benchmark ──────────────────────────────────────────────────────────
    public static class BenchOut {
        public long bfUs; public long kdUs; public long hnswUs; public int n;
    }

    public synchronized BenchOut benchmark(float[] q, int k, String metric) {
        DistFn dfn = DistanceMetrics.get(metric);
        BenchOut b = new BenchOut();
        b.n = store.size();

        long t;
        t = System.nanoTime(); bf.knn(q, k, dfn);   b.bfUs   = (System.nanoTime()-t)/1000;
        t = System.nanoTime(); kdt.knn(q, k, dfn);  b.kdUs   = (System.nanoTime()-t)/1000;
        t = System.nanoTime(); hnsw.knn(q, k, 50, dfn); b.hnswUs = (System.nanoTime()-t)/1000;
        return b;
    }

    // ── Accessors ──────────────────────────────────────────────────────────
    public synchronized List<VectorItem> all() { return new ArrayList<>(store.values()); }

    public synchronized HNSW.GraphInfo hnswInfo() { return hnsw.getInfo(); }

    public synchronized int size() { return store.size(); }
}