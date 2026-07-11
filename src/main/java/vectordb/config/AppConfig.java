package vectordb.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import vectordb.ollama.OllamaClient;

@Configuration
@EnableScheduling   // needed for RateLimiter's @Scheduled reset
public class AppConfig {

    @Bean
    public OllamaClient ollamaClient() {
        OllamaClient client = new OllamaClient();
        boolean up = client.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println("Ollama: " + (up ? "ONLINE" : "OFFLINE"));
        return client;
    }
}