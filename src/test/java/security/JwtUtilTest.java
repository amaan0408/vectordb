package security;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for JWT token generation and validation.
 * No Spring context — tests the pure utility logic.
 */
class JwtUtilTest {

    private JwtUtil jwtUtil;

    @BeforeEach
    void setUp() {
        jwtUtil = new JwtUtil();
    }

    @Test
    void generateToken_returnsNonNullToken() {
        String token = jwtUtil.generateToken("amaan", "USER");
        assertNotNull(token);
        assertFalse(token.isBlank());
    }

    @Test
    void generateToken_hasThreeParts() {
        // JWT format is header.payload.signature
        String token = jwtUtil.generateToken("amaan", "USER");
        String[] parts = token.split("\\.");
        assertEquals(3, parts.length, "JWT must have exactly 3 dot-separated parts");
    }

    @Test
    void extractUsername_returnsCorrectUsername() {
        String token = jwtUtil.generateToken("amaan", "USER");
        assertEquals("amaan", jwtUtil.extractUsername(token));
    }

    @Test
    void extractRole_returnsCorrectRole() {
        String token = jwtUtil.generateToken("amaan", "ADMIN");
        assertEquals("ADMIN", jwtUtil.extractRole(token));
    }

    @Test
    void isTokenValid_validToken_returnsTrue() {
        String token = jwtUtil.generateToken("amaan", "USER");
        assertTrue(jwtUtil.isTokenValid(token));
    }

    @Test
    void isTokenValid_tamperedToken_returnsFalse() {
        String token = jwtUtil.generateToken("amaan", "USER");
        String tampered = token + "corrupted";
        assertFalse(jwtUtil.isTokenValid(tampered));
    }

    @Test
    void isTokenValid_randomString_returnsFalse() {
        assertFalse(jwtUtil.isTokenValid("not.a.token"));
    }

    @Test
    void isTokenValid_emptyString_returnsFalse() {
        assertFalse(jwtUtil.isTokenValid(""));
    }

    @Test
    void differentUsers_producesDifferentTokens() {
        String t1 = jwtUtil.generateToken("amaan", "USER");
        String t2 = jwtUtil.generateToken("admin", "ADMIN");
        assertNotEquals(t1, t2);
    }

    @Test
    void extractUsername_adminUser_correct() {
        String token = jwtUtil.generateToken("admin", "ADMIN");
        assertEquals("admin", jwtUtil.extractUsername(token));
        assertEquals("ADMIN", jwtUtil.extractRole(token));
    }
}
