package vectordb.model;

/**
 * Equivalent to C++ struct VectorItem.
 */
public class VectorItem {
    public int id;
    public String metadata;
    public String category;
    public float[] emb;

    public VectorItem() {}

    public VectorItem(int id, String metadata, String category, float[] emb) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.emb = emb;
    }
}

