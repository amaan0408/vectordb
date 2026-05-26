package vectordb.server;

import com.google.gson.*;
import io.javalin.Javalin;
import io.javalin.http.Context;
import vectordb.db.DocumentDB;
import vectordb.db.VectorDB;
import vectordb.distance.DistanceMetrics;
import vectordb.index.HNSW;
import vectordb.model.DocItem;
import vectordb.model.VectorItem;
import vectordb.ollama.OllamaClient;
import vectordb.util.TextChunker;

import java.util.*;

/**
 * Registers all HTTP routes on a Javalin instance.
 * Equivalent to the entire httplib route-registration block in C++ main().
 */
public class Routes {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final int DIMS = 16;

    public static void register(Javalin app, VectorDB db, DocumentDB docDB, OllamaClient ollama) {

        // ── CORS for every response ──────────────────────────────────────
        app.before(ctx -> {
            ctx.header("Access-Control-Allow-Origin",  "*");
            ctx.header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            ctx.header("Access-Control-Allow-Headers", "Content-Type");
        });
        app.options("/*", ctx -> ctx.status(204));

        // ════════════════════════════════════════════════════════════════
        //  DEMO VECTOR ENDPOINTS
        // ════════════════════════════════════════════════════════════════

        // GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw
        app.get("/search", ctx -> {
            float[] q = parseVec(ctx.queryParam("v"));
            if (q == null || q.length != DIMS) {
                ctx.status(400).json(Map.of("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = intParam(ctx, "k", 5);
            String metric = defStr(ctx.queryParam("metric"), "cosine");
            String algo   = defStr(ctx.queryParam("algo"),   "hnsw");

            VectorDB.SearchOut out = db.search(q, k, metric, algo);

            List<Map<String, Object>> results = new ArrayList<>();
            for (VectorDB.Hit h : out.hits) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",        h.id);
                m.put("metadata",  h.meta);
                m.put("category",  h.cat);
                m.put("distance",  h.dist);
                m.put("embedding", h.emb);
                results.add(m);
            }
            ctx.json(Map.of(
                    "results",   results,
                    "latencyUs", out.latencyUs,
                    "algo",      out.algo,
                    "metric",    out.metric
            ));
        });

        // POST /insert  {"metadata":"...","category":"...","embedding":[...]}
        app.post("/insert", ctx -> {
            JsonObject body = JsonParser.parseString(ctx.body()).getAsJsonObject();
            String meta = strField(body, "metadata");
            String cat  = strField(body, "category");
            float[] emb = floatArray(body, "embedding");

            if (meta == null || emb == null || emb.length != DIMS) {
                ctx.status(400).json(Map.of("error", "invalid body"));
                return;
            }
            int id = db.insert(meta, cat == null ? "" : cat, emb, DistanceMetrics.get("cosine"));
            ctx.json(Map.of("id", id));
        });

        // DELETE /delete/:id
        app.delete("/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            ctx.json(Map.of("ok", db.remove(id)));
        });

        // GET /items
        app.get("/items", ctx -> {
            List<Map<String, Object>> list = new ArrayList<>();
            for (VectorItem v : db.all()) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",        v.id);
                m.put("metadata",  v.metadata);
                m.put("category",  v.category);
                m.put("embedding", v.emb);
                list.add(m);
            }
            ctx.json(list);
        });

        // GET /benchmark?v=...&k=5&metric=cosine
        app.get("/benchmark", ctx -> {
            float[] q = parseVec(ctx.queryParam("v"));
            if (q == null || q.length != DIMS) {
                ctx.status(400).json(Map.of("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = intParam(ctx, "k", 5);
            String metric = defStr(ctx.queryParam("metric"), "cosine");
            VectorDB.BenchOut b = db.benchmark(q, k, metric);
            ctx.json(Map.of(
                    "bruteforceUs", b.bfUs,
                    "kdtreeUs",     b.kdUs,
                    "hnswUs",       b.hnswUs,
                    "itemCount",    b.n
            ));
        });

        // GET /hnsw-info
        app.get("/hnsw-info", ctx -> {
            HNSW.GraphInfo gi = db.hnswInfo();
            ctx.json(Map.of(
                    "topLayer",      gi.topLayer,
                    "nodeCount",     gi.nodeCount,
                    "nodesPerLayer", gi.nodesPerLayer,
                    "edgesPerLayer", gi.edgesPerLayer,
                    "nodes",         gi.nodes,
                    "edges",         gi.edges
            ));
        });

        // GET /stats
        app.get("/stats", ctx -> ctx.json(Map.of(
                "count",      db.size(),
                "dims",       DIMS,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics",    List.of("euclidean", "cosine", "manhattan")
        )));

        // ════════════════════════════════════════════════════════════════
        //  DOCUMENT + RAG ENDPOINTS
        // ════════════════════════════════════════════════════════════════

        // POST /doc/insert  {"title":"...","text":"..."}
        app.post("/doc/insert", ctx -> {
            JsonObject body = JsonParser.parseString(ctx.body()).getAsJsonObject();
            String title = strField(body, "title");
            String text  = strField(body, "text");

            if (title == null || title.isEmpty() || text == null || text.isEmpty()) {
                ctx.status(400).json(Map.of("error", "need title and text"));
                return;
            }

            List<String> chunks = TextChunker.chunk(text);
            List<Integer> ids   = new ArrayList<>();

            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb == null) {
                    ctx.status(503).json(Map.of("error",
                            "Ollama unavailable. Install from https://ollama.com then run: " +
                                    "ollama pull nomic-embed-text && ollama pull llama3.2"));
                    return;
                }
                String chunkTitle = chunks.size() > 1
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            ctx.json(Map.of(
                    "ids",    ids,
                    "chunks", chunks.size(),
                    "dims",   docDB.getDims()
            ));
        });

        // DELETE /doc/delete/:id
        app.delete("/doc/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            ctx.json(Map.of("ok", docDB.remove(id)));
        });

        // GET /doc/list
        app.get("/doc/list", ctx -> {
            List<Map<String, Object>> list = new ArrayList<>();
            for (DocItem d : docDB.all()) {
                String preview = d.text.length() > 120
                        ? d.text.substring(0, 120) + "…"
                        : d.text;
                int words = d.text.split("\\s+").length;
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",      d.id);
                m.put("title",   d.title);
                m.put("preview", preview);
                m.put("words",   words);
                list.add(m);
            }
            ctx.json(list);
        });

        // POST /doc/search  {"question":"...","k":3}
        app.post("/doc/search", ctx -> {
            JsonObject body = JsonParser.parseString(ctx.body()).getAsJsonObject();
            String question = strField(body, "question");
            int k = body.has("k") ? body.get("k").getAsInt() : 3;

            if (question == null || question.isEmpty()) {
                ctx.status(400).json(Map.of("error", "need question"));
                return;
            }
            float[] qEmb = ollama.embed(question);
            if (qEmb == null) {
                ctx.status(503).json(Map.of("error", "Ollama unavailable"));
                return;
            }
            List<DocumentDB.DocItemWithDist> hits = docDB.search(qEmb, k);
            List<Map<String, Object>> contexts = new ArrayList<>();
            for (DocumentDB.DocItemWithDist h : hits) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",       h.item.id);
                m.put("title",    h.item.title);
                m.put("distance", h.dist);
                contexts.add(m);
            }
            ctx.json(Map.of("contexts", contexts));
        });

        // POST /doc/ask  {"question":"...","k":3}  — full RAG pipeline
        app.post("/doc/ask", ctx -> {
            JsonObject body = JsonParser.parseString(ctx.body()).getAsJsonObject();
            String question = strField(body, "question");
            int k = body.has("k") ? body.get("k").getAsInt() : 3;

            if (question == null || question.isEmpty()) {
                ctx.status(400).json(Map.of("error", "need question"));
                return;
            }

            // Step 1: embed the question
            float[] qEmb = ollama.embed(question);
            if (qEmb == null) {
                ctx.status(503).json(Map.of("error", "Ollama unavailable"));
                return;
            }

            // Step 2: retrieve top-k relevant chunks
            List<DocumentDB.DocItemWithDist> hits = docDB.search(qEmb, k);

            // Step 3: build prompt — same wording as C++
            StringBuilder ctx2 = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                ctx2.append("[").append(i + 1).append("] ")
                        .append(hits.get(i).item.title).append(":\n")
                        .append(hits.get(i).item.text).append("\n\n");
            }
            String prompt =
                    "You are a helpful assistant. Answer the user's question directly. " +
                            "Use the provided context if it contains relevant information. " +
                            "If it doesn't, just use your own general knowledge. " +
                            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like " +
                            "'the context doesn't mention'. Just answer the question naturally.\n\n" +
                            "Context:\n" + ctx2 +
                            "Question: " + question + "\n\nAnswer:";

            // Step 4: generate answer
            String answer = ollama.generate(prompt);

            // Step 5: build response
            List<Map<String, Object>> contexts = new ArrayList<>();
            for (DocumentDB.DocItemWithDist h : hits) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("id",       h.item.id);
                m.put("title",    h.item.title);
                m.put("text",     h.item.text);
                m.put("distance", h.dist);
                contexts.add(m);
            }
            ctx.json(Map.of(
                    "answer",   answer,
                    "model",    ollama.genModel,
                    "contexts", contexts,
                    "docCount", docDB.size()
            ));
        });

        // GET /status
        app.get("/status", ctx -> {
            boolean up = ollama.isAvailable();
            ctx.json(Map.of(
                    "ollamaAvailable", up,
                    "embedModel",      ollama.embedModel,
                    "genModel",        ollama.genModel,
                    "docCount",        docDB.size(),
                    "docDims",         docDB.getDims(),
                    "demoDims",        DIMS,
                    "demoCount",       db.size()
            ));
        });

        // ── Serve index.html ────────────────────────────────────────────
        app.get("/", ctx -> {
            try (var in = Routes.class.getResourceAsStream("/index.html")) {
                if (in == null) { ctx.status(404).result("index.html not found"); return; }
                ctx.html(new String(in.readAllBytes()));
            } catch (Exception e) {
                ctx.status(500).result("Error reading index.html");
            }
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return null;
        String[] parts = s.split(",");
        float[] v = new float[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
        } catch (NumberFormatException e) { return null; }
        return v;
    }

    private static int intParam(Context ctx, String name, int def) {
        try { return Integer.parseInt(ctx.queryParam(name)); } catch (Exception e) { return def; }
    }

    private static String defStr(String val, String def) {
        return (val == null || val.isEmpty()) ? def : val;
    }

    private static String strField(JsonObject obj, String key) {
        if (!obj.has(key)) return null;
        JsonElement el = obj.get(key);
        return el.isJsonNull() ? null : el.getAsString();
    }

    private static float[] floatArray(JsonObject obj, String key) {
        if (!obj.has(key)) return null;
        JsonArray arr = obj.getAsJsonArray(key);
        float[] v = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) v[i] = arr.get(i).getAsFloat();
        return v;
    }
}