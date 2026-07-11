package vectordb.server;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import vectordb.db.VectorDBRegistry;
import vectordb.repository.UserRepository;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/admin")
@CrossOrigin(origins = "*")
public class AdminController {

    private final VectorDBRegistry registry;
    private final UserRepository userRepo;

    public AdminController(VectorDBRegistry registry, UserRepository userRepo) {
        this.registry = registry;
        this.userRepo = userRepo;
    }

    @GetMapping("/users")
    @PreAuthorize("hasRole('ADMIN')")
    public List<String> listUsers() {
        return userRepo.findAll().stream().map(u -> u.getUsername()).toList();
    }

    @GetMapping("/stats")
    @PreAuthorize("hasRole('ADMIN')")
    public Map<String, Object> globalStats() {
        return Map.of(
                "totalUsers", registry.userCount(),
                "totalVectors", registry.totalVectorCount(),
                "activeUserDbs", registry.getAllUsernames()
        );
    }
}