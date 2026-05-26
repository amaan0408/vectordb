package vectordb.index;

import vectordb.distance.DistFn;
import vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Binary space-partitioning KD-Tree.
 * Direct translation of C++ class KDTree.
 * Works well for low dimensions (≤ 20D); degrades toward brute-force at 768D.
 */
public class KDTree {

    // ── Inner node ─────────────────────────────────────────────────────────
    private static class Node {
        VectorItem item;
        Node left, right;
        Node(VectorItem v) { this.item = v; }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    // ── Insert ─────────────────────────────────────────────────────────────
    public synchronized void insert(VectorItem v) {
        root = ins(root, v, 0);
    }

    private Node ins(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int ax = depth % dims;
        if (v.emb[ax] < n.item.emb[ax]) n.left  = ins(n.left,  v, depth + 1);
        else                             n.right = ins(n.right, v, depth + 1);
        return n;
    }

    // ── KNN ────────────────────────────────────────────────────────────────
    /**
     * Returns up to k (distance, id) pairs sorted ascending.
     * Uses a max-heap internally to maintain the k best candidates.
     */
    public synchronized List<float[]> knn(float[] q, int k, DistFn dist) {
        // Max-heap: we want the furthest of the k-best at the top so we can prune
        PriorityQueue<float[]> heap =
                new PriorityQueue<>(k + 1, (a, b) -> Float.compare(b[0], a[0]));
        knn(root, q, k, 0, dist, heap);

        List<float[]> result = new ArrayList<>(heap);
        result.sort((a, b) -> Float.compare(a[0], b[0]));
        return result;
    }

    private void knn(Node n, float[] q, int k, int depth, DistFn dist,
                     PriorityQueue<float[]> heap) {
        if (n == null) return;

        float dn = dist.compute(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek()[0]) {
            heap.offer(new float[]{dn, n.item.id});
            if (heap.size() > k) heap.poll();
        }

        int ax = depth % dims;
        float diff = q[ax] - n.item.emb[ax];
        Node closer  = diff < 0 ? n.left  : n.right;
        Node farther = diff < 0 ? n.right : n.left;

        knn(closer, q, k, depth + 1, dist, heap);

        // Prune: only search farther side if it could contain a closer point
        if (heap.size() < k || Math.abs(diff) < heap.peek()[0]) {
            knn(farther, q, k, depth + 1, dist, heap);
        }
    }

    // ── Rebuild (needed after delete) ──────────────────────────────────────
    public synchronized void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) root = ins(root, v, 0);
    }
}