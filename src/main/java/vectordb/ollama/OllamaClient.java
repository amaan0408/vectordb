package vectordb.ollama;

import com.google.gson.*;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * HTTP client for local Ollama — replaces C++ class OllamaClient.
 *
 * Requires:
 *   ollama pull nomic-embed-text
 *   ollama pull llama3.2
 */
public class OllamaClient {

    private static final String BASE = "http://127.0.0.1:11434";

    public String embedModel = "nomic-embed-text";
    public String genModel   = "llama3.2";

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();

    private final Gson gson = new Gson();

    // ── Availability check ────────────────────────────────────────────────
    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(BASE + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET().build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ── Embed ──────────────────────────────────────────────────────────────
    /**
     * Calls /api/embeddings and returns the float[] vector.
     * Returns null if Ollama is unavailable.
     * Equivalent to C++ OllamaClient::embed().
     */
    public float[] embed(String text) {
        try {
            String body = gson.toJson(new EmbedRequest(embedModel, text));
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(BASE + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return null;

            JsonObject json = JsonParser.parseString(res.body()).getAsJsonObject();
            JsonArray arr = json.getAsJsonArray("embedding");
            if (arr == null) return null;

            float[] emb = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++)
            emb[i] = arr.get(i).getAsFloat();
            return emb;

        } catch (Exception e) {
            return null;
        }
    }

    // ── Generate ──────────────────────────────────────────────────────────
    /**
     * Calls /api/generate (stream=false) and returns the answer string.
     * Equivalent to C++ OllamaClient::generate().
     */
    public String generate(String prompt) {
        try {
            String body = gson.toJson(new GenerateRequest(genModel, prompt, false));
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(BASE + "/api/generate"))
                    .timeout(Duration.ofSeconds(180))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200)
                return "ERROR: Ollama unavailable. Run: ollama serve";

            JsonObject json = JsonParser.parseString(res.body()).getAsJsonObject();
            JsonElement el = json.get("response");
            return el != null ? el.getAsString() : "ERROR: empty response";

        } catch (Exception e) {
            return "ERROR: " + e.getMessage();
        }
    }

    // ── Request POJOs for Gson serialisation ──────────────────────────────
    private static class EmbedRequest {
        String model; String prompt;
        EmbedRequest(String model, String prompt) { this.model=model; this.prompt=prompt; }
    }

    private static class GenerateRequest {
        String model; String prompt; boolean stream;
        GenerateRequest(String model, String prompt, boolean stream) {
            this.model=model; this.prompt=prompt; this.stream=stream;
        }
    }
}