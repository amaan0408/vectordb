package vectordb.model;

/**
 * Equivalent to C++ struct DocItem — stores a document chunk with its embedding.
 */
public class DocItem {
    public int id;
    public String title;
    public String text;
    public float[] emb;

    public DocItem() {}

    public DocItem(int id, String title, String text, float[] emb) {
        this.id = id;
        this.title = title;
        this.text = text;
        this.emb = emb;
    }
}

