package vectordb.distance;

/**
 * Equivalent to C++ typedef: using DistFn = std::function<float(...)>
 */
@FunctionalInterface
public interface DistFn {
    float compute(float[] a, float[] b);
}
