package vectordb.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits long text into overlapping word-chunks.
 * Direct translation of C++ chunkText().
 */
public class TextChunker {

    /**
     * @param text         raw input text
     * @param chunkWords   target chunk size in words (default 250)
     * @param overlapWords overlap between adjacent chunks (default 30)
     */
    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) return List.of();
        if (words.length <= chunkWords) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;

        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(' ');
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    /** Convenience overload with default sizes matching the C++ original. */
    public static List<String> chunk(String text) {
        return chunk(text, 250, 30);
    }
}