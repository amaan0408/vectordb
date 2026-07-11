package vectordb.auth;

import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;
import vectordb.entity.User;
import vectordb.repository.UserRepository;
import vectordb.security.JwtUtil;

import java.util.Map;

@RestController
@RequestMapping("/auth")
@CrossOrigin(origins = "*")
public class AuthController {

    private final UserRepository userRepo;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    public AuthController(UserRepository userRepo,
                          PasswordEncoder passwordEncoder,
                          JwtUtil jwtUtil) {
        this.userRepo = userRepo;
        this.passwordEncoder = passwordEncoder;
        this.jwtUtil = jwtUtil;
    }

    // ── POST /auth/register ────────────────────────────────────────────
    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody AuthRequest req) {
        if (req.getUsername() == null || req.getUsername().isBlank() ||
                req.getPassword() == null || req.getPassword().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "username and password required"));
        }

        if (userRepo.existsByUsername(req.getUsername())) {
            return ResponseEntity.badRequest().body(Map.of("error", "username already taken"));
        }

        String hashed = passwordEncoder.encode(req.getPassword());
        User user = new User(req.getUsername(), hashed, "USER");
        userRepo.save(user);

        String token = jwtUtil.generateToken(user.getUsername(), user.getRole());
        return ResponseEntity.ok(new AuthResponse(token, user.getUsername(), user.getRole()));
    }

    // ── POST /auth/login ───────────────────────────────────────────────
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody AuthRequest req) {
        var userOpt = userRepo.findByUsername(req.getUsername());

        if (userOpt.isEmpty() ||
                !passwordEncoder.matches(req.getPassword(), userOpt.get().getPassword())) {
            return ResponseEntity.status(401).body(Map.of("error", "invalid username or password"));
        }

        User user = userOpt.get();
        String token = jwtUtil.generateToken(user.getUsername(), user.getRole());
        return ResponseEntity.ok(new AuthResponse(token, user.getUsername(), user.getRole()));
    }

    // ── POST /auth/make-admin (dev helper — remove in real production) ──
    @PostMapping("/make-admin/{username}")
    public ResponseEntity<?> makeAdmin(@PathVariable String username) {
        var userOpt = userRepo.findByUsername(username);
        if (userOpt.isEmpty()) {
            return ResponseEntity.status(404).body(Map.of("error", "user not found"));
        }
        User user = userOpt.get();
        user.setRole("ADMIN");
        userRepo.save(user);
        return ResponseEntity.ok(Map.of("message", username + " is now ADMIN"));
    }
}