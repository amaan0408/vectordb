package vectordb.server;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import vectordb.db.VectorDB;
import vectordb.db.VectorDBRegistry;
import vectordb.distance.DistanceMetrics;
import vectordb.index.HNSW;
import vectordb.model.VectorItem;

import java.util.*;

@RestController
@CrossOrigin(origins = "*")
public class VectorController {

    private final VectorDBRegistry registry;

    public VectorController(VectorDBRegistry registry) {
        this.registry = registry;
    }

    @GetMapping("/search")
    public Map<String, Object> search(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric,
            @RequestParam(defaultValue = "hnsw") String algo,
            Authentication auth) {

        VectorDB db = registry.getForUser(auth.getName());

        float[] q = parseVec(v);
        if (q == null || q.length != 16) {
            return Map.of("error", "need 16D vector");
        }

        VectorDB.SearchOut out = db.search(q, k, metric, algo);

        List<Map<String, Object>> results = new ArrayList<>();
        for (VectorDB.Hit h : out.hits) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", h.id);
            m.put("metadata", h.meta);
            m.put("category", h.cat);
            m.put("distance", h.dist);
            m.put("embedding", h.emb);
            results.add(m);
        }

        return Map.of(
                "results", results,
                "latencyUs", out.latencyUs,
                "algo", out.algo,
                "metric", out.metric
        );
    }

    @PostMapping("/insert")
    public Map<String, Object> insert(@RequestBody String body, Authentication auth) {
        VectorDB db = registry.getForUser(auth.getName());

        JsonObject json = JsonParser.parseString(body).getAsJsonObject();
        String meta = strField(json, "metadata");
        String cat = strField(json, "category");
        float[] emb = floatArray(json, "embedding");

        if (meta == null || emb == null || emb.length != 16) {
            return Map.of("error", "invalid body");
        }

        int id = db.insert(meta, cat == null ? "" : cat, emb, DistanceMetrics.get("cosine"));
        return Map.of("id", id);
    }

    @DeleteMapping("/delete/{id}")
    public Map<String, Object> delete(@PathVariable int id, Authentication auth) {
        VectorDB db = registry.getForUser(auth.getName());
        return Map.of("ok", db.remove(id));
    }

    @GetMapping("/items")
    public List<Map<String, Object>> items(Authentication auth) {
        VectorDB db = registry.getForUser(auth.getName());
        List<Map<String, Object>> list = new ArrayList<>();
        for (VectorItem v : db.all()) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", v.id);
            m.put("metadata", v.metadata);
            m.put("category", v.category);
            m.put("embedding", v.emb);
            list.add(m);
        }
        return list;
    }

    @GetMapping("/benchmark")
    public Map<String, Object> benchmark(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric,
            Authentication auth) {

        VectorDB db = registry.getForUser(auth.getName());
        float[] q = parseVec(v);
        if (q == null || q.length != 16) {
            return Map.of("error", "need 16D vector");
        }

        VectorDB.BenchOut b = db.benchmark(q, k, metric);
        return Map.of(
                "bruteforceUs", b.bfUs,
                "kdtreeUs", b.kdUs,
                "hnswUs", b.hnswUs,
                "itemCount", b.n
        );
    }

    @GetMapping("/hnsw-info")
    public Map<String, Object> hnswInfo(Authentication auth) {
        VectorDB db = registry.getForUser(auth.getName());
        HNSW.GraphInfo gi = db.hnswInfo();
        return Map.of(
                "topLayer", gi.topLayer,
                "nodeCount", gi.nodeCount,
                "nodesPerLayer", gi.nodesPerLayer,
                "edgesPerLayer", gi.edgesPerLayer,
                "nodes", gi.nodes,
                "edges", gi.edges
        );
    }

    @GetMapping("/stats")
    public Map<String, Object> stats(Authentication auth) {
        VectorDB db = registry.getForUser(auth.getName());
        return Map.of(
                "count", db.size(),
                "dims", 16,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics", List.of("euclidean", "cosine", "manhattan")
        );
    }

    private float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return null;
        String[] parts = s.split(",");
        float[] v = new float[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
        } catch (NumberFormatException e) { return null; }
        return v;
    }

    private String strField(JsonObject obj, String key) {
        if (!obj.has(key)) return null;
        var el = obj.get(key);
        return el.isJsonNull() ? null : el.getAsString();
    }

    private float[] floatArray(JsonObject obj, String key) {
        if (!obj.has(key)) return null;
        var arr = obj.getAsJsonArray(key);
        float[] v = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) v[i] = arr.get(i).getAsFloat();
        return v;
    }
}