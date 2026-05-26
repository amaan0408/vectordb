package vectordb;

import io.javalin.Javalin;
import vectordb.db.DocumentDB;
import vectordb.db.VectorDB;
import vectordb.ollama.OllamaClient;
import vectordb.server.Routes;
import vectordb.util.DemoLoader;

/**
 * Entry point — equivalent to C++ main().
 *
 * Build:  mvn package
 * Run:    java -jar target/vectordb.jar
 * Open:   http://localhost:8080
 */
public class Main {

    private static final int PORT = 8080;
    private static final int DIMS = 16;

    public static void main(String[] args) {

        // ── 1. Initialise databases ──────────────────────────────────────
        VectorDB   db     = new VectorDB(DIMS);
        DocumentDB docDB  = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        // ── 2. Load 20 demo vectors (same as C++ loadDemo) ───────────────
        DemoLoader.load(db);

        // ── 3. Check Ollama (non-fatal) ──────────────────────────────────
        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:" + PORT);
        System.out.println(db.size() + " demo vectors | " + DIMS
                + " dims | HNSW + KD-Tree + BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel
                    + "   gen model: " + ollama.genModel);
        }

        // ── 4. Start Javalin HTTP server ─────────────────────────────────
        Javalin app = Javalin.create(config -> {
            // Suppress Javalin's default startup banner to keep output clean
            config.bundledPlugins.enableCors(cors ->
                    cors.addRule(rule -> rule.anyHost())
            );
        });

        Routes.register(app, db, docDB, ollama);

        app.start(PORT);
    }
}