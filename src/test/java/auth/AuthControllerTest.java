package auth;

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

import java.util.Map;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.hamcrest.Matchers.*;

/**
 * Integration tests for /auth/register and /auth/login endpoints.
 * Starts the full Spring context with an H2 in-memory database.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AuthControllerTest {

    @Autowired MockMvc mockMvc;
    @Autowired UserRepository userRepo;
    @Autowired ObjectMapper objectMapper;

    @BeforeEach
    void cleanDb() {
        userRepo.deleteAll();
    }

    // ── Register ──────────────────────────────────────────────────────

    @Test
    void register_validCredentials_returns200WithToken() throws Exception {
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "username", "amaan",
                    "password", "test123"
                ))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token", not(emptyString())))
                .andExpect(jsonPath("$.username", is("amaan")))
                .andExpect(jsonPath("$.role", is("USER")));
    }

    @Test
    void register_duplicateUsername_returns400() throws Exception {
        String body = objectMapper.writeValueAsString(Map.of("username","amaan","password","test123"));

        mockMvc.perform(post("/auth/register").contentType(MediaType.APPLICATION_JSON).content(body))
               .andExpect(status().isOk());

        mockMvc.perform(post("/auth/register").contentType(MediaType.APPLICATION_JSON).content(body))
               .andExpect(status().isBadRequest())
               .andExpect(jsonPath("$.error", containsString("taken")));
    }

    @Test
    void register_missingUsername_returns400() throws Exception {
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("password", "test123"))))
                .andExpect(status().isBadRequest());
    }

    @Test
    void register_missingPassword_returns400() throws Exception {
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username", "amaan"))))
                .andExpect(status().isBadRequest());
    }

    @Test
    void register_blankUsername_returns400() throws Exception {
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","  ","password","test123"))))
                .andExpect(status().isBadRequest());
    }

    // ── Login ─────────────────────────────────────────────────────────

    @Test
    void login_correctCredentials_returns200WithToken() throws Exception {
        // First register
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","amaan","password","test123"))))
               .andExpect(status().isOk());

        // Then login
        mockMvc.perform(post("/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","amaan","password","test123"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token", not(emptyString())))
                .andExpect(jsonPath("$.username", is("amaan")));
    }

    @Test
    void login_wrongPassword_returns401() throws Exception {
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","amaan","password","test123"))))
               .andExpect(status().isOk());

        mockMvc.perform(post("/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","amaan","password","wrong"))))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.error", containsString("invalid")));
    }

    @Test
    void login_nonExistentUser_returns401() throws Exception {
        mockMvc.perform(post("/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("username","ghost","password","test123"))))
                .andExpect(status().isUnauthorized());
    }
}
