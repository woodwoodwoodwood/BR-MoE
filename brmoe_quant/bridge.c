#include "ggml.h"
#include <stdint.h>

// Use GGML's own type traits rather than reproducing its private struct in ctypes.
int brmoe_dequantize(int type, const void * src, float * dst, int64_t n) {
    const struct ggml_type_traits * traits = ggml_get_type_traits((enum ggml_type) type);
    if (!traits || !traits->to_float) return -1;
    traits->to_float(src, dst, n);
    return 0;
}
