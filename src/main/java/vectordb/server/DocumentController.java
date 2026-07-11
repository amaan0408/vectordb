package vectordb.server;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import vectordb.db.DocumentDB;
import vectordb.db.DocumentDBRegistry;
import vectordb.model.DocItem;
import vectordb.ollama.OllamaClient;
import vectordb.util.TextChunker;

import java.util.*;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/doc")
public class DocumentController {

    private final DocumentDBRegistry registry;
    private final OllamaClient ollama;

    public DocumentController(DocumentDBRegistry registry, OllamaClient ollama) {
        this.registry = registry;
        this.ollama = ollama;
    }

    @PostMapping("/insert")
    public Map<String, Object> insert(@RequestBody String body, Authentication auth) {
        DocumentDB docDB = registry.getForUser(auth.getName());

        JsonObject json = JsonParser.parseString(body).getAsJsonObject();
        String title = strField(json, "title");
        String text = strField(json, "text");

        if (title == null || title.isEmpty() || text == null || text.isEmpty()) {
            return Map.of("error", "need title and text");
        }

        List<String> chunks = TextChunker.chunk(text);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb == null) {
                return Map.of("error", "Ollama unavailable. Run: ollama pull nomic-embed-text");
            }
            String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        return Map.of("ids", ids, "chunks", chunks.size(), "dims", docDB.getDims());
    }

    @DeleteMapping("/delete/{id}")
    public Map<String, Object> delete(@PathVariable int id, Authentication auth) {
        DocumentDB docDB = registry.getForUser(auth.getName());
        return Map.of("ok", docDB.remove(id));
    }

    @GetMapping("/list")
    public List<Map<String, Object>> list(Authentication auth) {
        DocumentDB docDB = registry.getForUser(auth.getName());
        List<Map<String, Object>> result = new ArrayList<>();
        for (DocItem d : docDB.all()) {
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", d.id);
            m.put("title", d.title);
            m.put("preview", preview);
            m.put("words", d.text.split("\\s+").length);
            result.add(m);
        }
        return result;
    }

    @PostMapping("/search")
    public Map<String, Object> search(@RequestBody String body, Authentication auth) {
        DocumentDB docDB = registry.getForUser(auth.getName());

        JsonObject json = JsonParser.parseString(body).getAsJsonObject();
        String question = strField(json, "question");
        int k = json.has("k") ? json.get("k").getAsInt() : 3;

        if (question == null || question.isEmpty()) {
            return Map.of("error", "need question");
        }

        float[] qEmb = ollama.embed(question);
        if (qEmb == null) return Map.of("error", "Ollama unavailable");

        List<DocumentDB.DocItemWithDist> hits = docDB.search(qEmb, k);
        List<Map<String, Object>> contexts = new ArrayList<>();
        for (var h : hits) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", h.item.id);
            m.put("title", h.item.title);
            m.put("distance", h.dist);
            contexts.add(m);
        }
        return Map.of("contexts", contexts);
    }

    @PostMapping("/ask")
    public Map<String, Object> ask(@RequestBody String body, Authentication auth) {
        DocumentDB docDB = registry.getForUser(auth.getName());

        JsonObject json = JsonParser.parseString(body).getAsJsonObject();
        String question = strField(json, "question");
        int k = json.has("k") ? json.get("k").getAsInt() : 3;

        if (question == null || question.isEmpty()) {
            return Map.of("error", "need question");
        }

        float[] qEmb = ollama.embed(question);
        if (qEmb == null) return Map.of("error", "Ollama unavailable");

        List<DocumentDB.DocItemWithDist> hits = docDB.search(qEmb, k);

        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ")
                    .append(hits.get(i).item.title).append(":\n")
                    .append(hits.get(i).item.text).append("\n\n");
        }
        String prompt =
                "You are a helpful assistant. Answer the user's question directly. " +
                        "Use the provided context if relevant. Do NOT mention 'the context'.\n\n" +
                        "Context:\n" + ctx +
                        "Question: " + question + "\n\nAnswer:";

        String answer = ollama.generate(prompt);

        List<Map<String, Object>> contexts = new ArrayList<>();
        for (var h : hits) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", h.item.id);
            m.put("title", h.item.title);
            m.put("text", h.item.text);
            m.put("distance", h.dist);
            contexts.add(m);
        }

        return Map.of(
                "answer", answer,
                "model", ollama.genModel,
                "contexts", contexts,
                "docCount", docDB.size()
        );
    }

    private String strField(JsonObject obj, String key) {
        if (!obj.has(key)) return null;
        var el = obj.get(key);
        return el.isJsonNull() ? null : el.getAsString();
    }
}