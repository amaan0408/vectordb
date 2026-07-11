package vectordb.db;

import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class DocumentDBRegistry {

    private final Map<String, DocumentDB> userDocDbs = new ConcurrentHashMap<>();

    public DocumentDB getForUser(String username) {
        return userDocDbs.computeIfAbsent(username, u -> new DocumentDB());
    }
}