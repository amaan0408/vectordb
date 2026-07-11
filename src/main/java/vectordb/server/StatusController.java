package vectordb.server;

import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import vectordb.db.DocumentDBRegistry;
import vectordb.db.VectorDBRegistry;
import vectordb.ollama.OllamaClient;

import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
public class StatusController {

    private final VectorDBRegistry vectorRegistry;
    private final DocumentDBRegistry docRegistry;
    private final OllamaClient ollama;

    public StatusController(VectorDBRegistry vectorRegistry,
                            DocumentDBRegistry docRegistry,
                            OllamaClient ollama) {
        this.vectorRegistry = vectorRegistry;
        this.docRegistry = docRegistry;
        this.ollama = ollama;
    }

    @GetMapping("/status")
    public Map<String, Object> status(Authentication auth) {
        boolean up = ollama.isAvailable();
        String username = auth != null ? auth.getName() : "anonymous";
        return Map.of(
                "ollamaAvailable", up,
                "embedModel", ollama.embedModel,
                "genModel", ollama.genModel,
                "docCount", docRegistry.getForUser(username).size(),
                "docDims", docRegistry.getForUser(username).getDims(),
                "demoDims", 16,
                "demoCount", vectorRegistry.getForUser(username).size()
        );
    }
}