package vectordb.index;

import vectordb.distance.DistFn;
import vectordb.model.VectorItem;

import java.util.*;

/**
 * Hierarchical Navigable Small World graph.
 * Direct translation of the C++ HNSW class.
 * Same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
 *
 * Insert: O(log N)   Search: O(log N)
 */
public class HNSW {

    // ── Node ───────────────────────────────────────────────────────────────
    private static class Node {
        VectorItem item;
        int maxLayer;
        List<List<Integer>> nbrs; // per-layer neighbour lists

        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.nbrs = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++) nbrs.add(new ArrayList<>());
        }
    }

    // ── Fields ─────────────────────────────────────────────────────────────
    private final Map<Integer, Node> G = new HashMap<>();
    private final int M;          // max connections per layer (except layer 0)
    private final int M0;         // max connections at layer 0 = 2*M
    private final int efBuild;    // beam width during construction
    private final double mL;      // level-generation factor = 1/ln(M)
    private int topLayer = -1;
    private int entryPt  = -1;
    private final Random rng = new Random(42);

    public HNSW() { this(16, 200); }

    public HNSW(int M, int efBuild) {
        this.M       = M;
        this.M0      = 2 * M;
        this.efBuild = efBuild;
        this.mL      = 1.0 / Math.log(M);
    }

    // ── Random level generation ────────────────────────────────────────────
    private int randLevel() {
        return (int) Math.floor(-Math.log(rng.nextDouble()) * mL);
    }

    // ── searchLayer ────────────────────────────────────────────────────────
    /**
     * Greedy beam search on a single layer.
     * Returns list of (dist, id) sorted ascending — same as C++ version.
     */
    private List<float[]> searchLayer(float[] q, int ep, int ef, int layer, DistFn dist) {
        Set<Integer> visited = new HashSet<>();
        // min-heap for candidates (closest first)
        PriorityQueue<float[]> cands =
                new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        // max-heap for found set (furthest first, so we can evict)
        PriorityQueue<float[]> found =
                new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

        float d0 = dist.compute(q, G.get(ep).item.emb);
        visited.add(ep);
        cands.offer(new float[]{d0, ep});
        found.offer(new float[]{d0, ep});

        while (!cands.isEmpty()) {
            float[] cur = cands.poll();
            float cd = cur[0];
            int cid = (int) cur[1];

            if (found.size() >= ef && cd > found.peek()[0]) break;

            Node cNode = G.get(cid);
            if (cNode == null || layer >= cNode.nbrs.size()) continue;

            for (int nid : cNode.nbrs.get(layer)) {
                if (visited.contains(nid) || !G.containsKey(nid)) continue;
                visited.add(nid);
                float nd = dist.compute(q, G.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek()[0]) {
                    cands.offer(new float[]{nd, nid});
                    found.offer(new float[]{nd, nid});
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<float[]> res = new ArrayList<>(found);
        res.sort(Comparator.comparingDouble(a -> a[0]));
        return res;
    }

    // ── selectNbrs ─────────────────────────────────────────────────────────
    private List<Integer> selectNbrs(List<float[]> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
            r.add((int) cands.get(i)[1]);
        }
        return r;
    }

    // ── Insert ─────────────────────────────────────────────────────────────
    public synchronized void insert(VectorItem item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G.put(id, new Node(item, lvl));

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;

        // Greedy descent from topLayer down to lvl+1
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }

        // Insert at each layer from min(topLayer, lvl) down to 0
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<float[]> W = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);

            // Ensure neighbour list exists for this layer
            while (G.get(id).nbrs.size() <= lc) G.get(id).nbrs.add(new ArrayList<>());
            G.get(id).nbrs.set(lc, new ArrayList<>(sel));

            // Bidirectional connections
            for (int nid : sel) {
                Node nNode = G.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                nNode.nbrs.get(lc).add(id);

                // Prune if over-connected
                if (nNode.nbrs.get(lc).size() > maxM) {
                    List<float[]> ds = new ArrayList<>();
                    for (int c : nNode.nbrs.get(lc)) {
                        if (G.containsKey(c)) {
                            ds.add(new float[]{dist.compute(nNode.item.emb, G.get(c).item.emb), c});
                        }
                    }
                    ds.sort(Comparator.comparingDouble(a -> a[0]));
                    List<Integer> pruned = new ArrayList<>();
                    for (int i = 0; i < maxM && i < ds.size(); i++) pruned.add((int) ds.get(i)[1]);
                    nNode.nbrs.set(lc, pruned);
                }
            }

            if (!W.isEmpty()) ep = (int) W.get(0)[1];
        }

        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    // ── KNN Search ─────────────────────────────────────────────────────────
    public synchronized List<float[]> knn(float[] q, int k, int ef, DistFn dist) {
        if (entryPt == -1) return Collections.emptyList();

        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }

        List<float[]> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) W = W.subList(0, k);
        return new ArrayList<>(W);
    }

    // ── Remove ─────────────────────────────────────────────────────────────
    public synchronized void remove(int id) {
        if (!G.containsKey(id)) return;
        // Remove id from all neighbour lists
        for (Node nd : G.values()) {
            for (List<Integer> layer : nd.nbrs) layer.remove((Integer) id);
        }
        // Update entry point if needed
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : G.keySet()) {
                if (nid != id) { entryPt = nid; break; }
            }
        }
        G.remove(id);
    }

    // ── Graph info (for /hnsw-info endpoint) ──────────────────────────────
    public static class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();

        public static class NodeView {
            public int id; public String metadata; public String category; public int maxLyr;
        }
        public static class EdgeView {
            public int src; public int dst; public int lyr;
        }
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();
    }

    public synchronized GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer  = topLayer;
        gi.nodeCount = G.size();

        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) { gi.nodesPerLayer.add(0); gi.edgesPerLayer.add(0); }

        for (Map.Entry<Integer, Node> e : G.entrySet()) {
            int id = e.getKey(); Node nd = e.getValue();
            GraphInfo.NodeView nv = new GraphInfo.NodeView();
            nv.id = id; nv.metadata = nd.item.metadata;
            nv.category = nd.item.category; nv.maxLyr = nd.maxLayer;
            gi.nodes.add(nv);

            for (int lc = 0; lc <= nd.maxLayer && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            GraphInfo.EdgeView ev = new GraphInfo.EdgeView();
                            ev.src = id; ev.dst = nid; ev.lyr = lc;
                            gi.edges.add(ev);
                        }
                    }
                }
            }
        }
        return gi;
    }

    public synchronized int size() { return G.size(); }
}