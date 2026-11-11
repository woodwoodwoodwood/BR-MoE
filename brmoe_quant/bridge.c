#include "ggml.h"
#include <stdint.h>
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

// Use GGML's own type traits rather than reproducing its private struct in ctypes.
int brmoe_dequantize(int type, const void * src, float * dst, int64_t n) {
    const struct ggml_type_traits * traits = ggml_get_type_traits((enum ggml_type) type);
    if (!traits || !traits->to_float) return -1;
    traits->to_float(src, dst, n);
    return 0;
}

// Export the pinned upstream codebooks; GPU decoding must use identical tables.
const void * brmoe_quant_table(int type, int * bytes) {
    switch (type) {
        case 16: *bytes = sizeof(iq2xxs_grid); return iq2xxs_grid;
        case 17: *bytes = sizeof(iq2xs_grid); return iq2xs_grid;
        case 18: *bytes = sizeof(iq3xxs_grid); return iq3xxs_grid;
        case 20: case 23: *bytes = sizeof(kvalues_iq4nl); return kvalues_iq4nl;
        case 21: *bytes = sizeof(iq3s_grid); return iq3s_grid;
        case 22: *bytes = sizeof(iq2s_grid); return iq2s_grid;
        case -1: *bytes = sizeof(ksigns_iq2xs); return ksigns_iq2xs;
        default: *bytes = 0; return 0;
    }
}
