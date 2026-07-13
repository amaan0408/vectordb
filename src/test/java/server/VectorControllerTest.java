package vectordb.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import vectordb.repository.UserRepository;
import vectordb.security.JwtUtil;

import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Integration tests for vector search, insert, and delete endpoints.
 * Every request includes a valid JWT in the Authorization header.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class VectorControllerTest {

    @Autowired MockMvc mockMvc;
    @Autowired JwtUtil jwtUtil;
    @Autowired UserRepository userRepo;
    @Autowired ObjectMapper objectMapper;

    private String token;

    // 16D zero vector — valid shape, won't crash anything
    private static final List<Float> ZERO_VEC = List.of(
        0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
        0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f
    );
    private static final String ZERO_VEC_STR =
        "0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1";

    @BeforeEach
    void setUp() {
        // Generate a valid token — no need to hit /auth/register for every test
        token = jwtUtil.generateToken("testuser", "USER");
    }

    private String bearer() { return "Bearer " + token; }

    // ── Auth guard ────────────────────────────────────────────────────

    @Test
    void items_withoutToken_returns401() throws Exception {
        mockMvc.perform(get("/items"))
               .andExpect(status().isUnauthorized());
    }

    @Test
    void search_withoutToken_returns401() throws Exception {
        mockMvc.perform(get("/search").param("v", ZERO_VEC_STR))
               .andExpect(status().isUnauthorized());
    }

    // ── GET /items ────────────────────────────────────────────────────

    @Test
    void items_withValidToken_returns200() throws Exception {
        mockMvc.perform(get("/items").header("Authorization", bearer()))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$", isA(List.class)));
    }

    @Test
    void items_newUser_hasDemoVectors() throws Exception {
        // VectorDBRegistry seeds demo data for every new user
        mockMvc.perform(get("/items").header("Authorization", bearer()))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$", hasSize(greaterThan(0))));
    }

    // ── GET /search ───────────────────────────────────────────────────

    @Test
    void search_validQuery_returnsResults() throws Exception {
        mockMvc.perform(get("/search")
                .header("Authorization", bearer())
                .param("v", ZERO_VEC_STR)
                .param("k", "3")
                .param("metric", "cosine")
                .param("algo", "hnsw"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.results", hasSize(lessThanOrEqualTo(3))))
               .andExpect(jsonPath("$.latencyUs", greaterThanOrEqualTo(0)));
    }

    @Test
    void search_bruteforce_returnsResults() throws Exception {
        mockMvc.perform(get("/search")
                .header("Authorization", bearer())
                .param("v", ZERO_VEC_STR)
                .param("k", "2")
                .param("metric", "euclidean")
                .param("algo", "bruteforce"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.algo", is("bruteforce")));
    }

    @Test
    void search_wrongDimension_returns200WithError() throws Exception {
        // 3D vector against a 16D index
        mockMvc.perform(get("/search")
                .header("Authorization", bearer())
                .param("v", "0.1,0.2,0.3"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.error", notNullValue()));
    }

    // ── POST /insert ──────────────────────────────────────────────────

    @Test
    void insert_validVector_returnsId() throws Exception {
        Map<String, Object> body = Map.of(
            "metadata",  "Test vector",
            "category",  "cs",
            "embedding", ZERO_VEC
        );
        mockMvc.perform(post("/insert")
                .header("Authorization", bearer())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(body)))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.id", greaterThan(0)));
    }

    @Test
    void insert_missingMetadata_returnsError() throws Exception {
        Map<String, Object> body = Map.of("category","cs","embedding", ZERO_VEC);
        mockMvc.perform(post("/insert")
                .header("Authorization", bearer())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(body)))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.error", notNullValue()));
    }

    // ── DELETE /delete/{id} ───────────────────────────────────────────

    @Test
    void delete_nonExistentId_returnsOkFalse() throws Exception {
        mockMvc.perform(delete("/delete/99999")
                .header("Authorization", bearer()))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.ok", is(false)));
    }

    // ── GET /stats ────────────────────────────────────────────────────

    @Test
    void stats_withToken_returnsCorrectStructure() throws Exception {
        mockMvc.perform(get("/stats").header("Authorization", bearer()))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.dims", is(16)))
               .andExpect(jsonPath("$.algorithms", hasSize(3)))
               .andExpect(jsonPath("$.metrics", hasSize(3)));
    }

    // ── GET /benchmark ────────────────────────────────────────────────

    @Test
    void benchmark_returnsAllThreeLatencies() throws Exception {
        mockMvc.perform(get("/benchmark")
                .header("Authorization", bearer())
                .param("v", ZERO_VEC_STR)
                .param("k", "3")
                .param("metric", "cosine"))
               .andExpect(status().isOk())
               .andExpect(jsonPath("$.bruteforceUs", greaterThanOrEqualTo(0)))
               .andExpect(jsonPath("$.kdtreeUs",     greaterThanOrEqualTo(0)))
               .andExpect(jsonPath("$.hnswUs",       greaterThanOrEqualTo(0)));
    }

    // ── Tenant isolation ──────────────────────────────────────────────

    @Test
    void insert_twoUsers_doNotShareData() throws Exception {
        String tokenA = jwtUtil.generateToken("userA", "USER");
        String tokenB = jwtUtil.generateToken("userB", "USER");

        // userA inserts a vector
        Map<String, Object> body = Map.of(
            "metadata","UserA exclusive","category","cs","embedding", ZERO_VEC
        );
        mockMvc.perform(post("/insert")
                .header("Authorization", "Bearer " + tokenA)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(body)))
               .andExpect(status().isOk());

        // userB's items should NOT contain userA's vector
        String itemsB = mockMvc.perform(get("/items")
                .header("Authorization", "Bearer " + tokenB))
               .andExpect(status().isOk())
               .andReturn().getResponse().getContentAsString();

        assertFalse(itemsB.contains("UserA exclusive"),
            "userB should not see userA's vectors");
    }
}
