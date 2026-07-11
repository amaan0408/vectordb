package vectordb.db;

import org.springframework.stereotype.Service;
import vectordb.util.DemoLoader;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Maps each username to their own isolated VectorDB instance.
 * This is what makes the application multi-tenant.
 */
@Service
public class VectorDBRegistry {

    private final Map<String, VectorDB> userDbs = new ConcurrentHashMap<>();

    public VectorDB getForUser(String username) {
        return userDbs.computeIfAbsent(username, u -> {
            VectorDB db = new VectorDB(16);
            DemoLoader.load(db);   // every new user starts with demo data
            return db;
        });
    }

    public Set<String> getAllUsernames() {
        return userDbs.keySet();
    }

    public int userCount() {
        return userDbs.size();
    }

    public int totalVectorCount() {
        return userDbs.values().stream().mapToInt(VectorDB::size).sum();
    }
}