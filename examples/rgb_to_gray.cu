// Pure elementwise, computer-vision edition. One output pixel per thread, reads
// three planar input channels (R, G, B), no loops, no shared memory, no
// cross-lane communication. Classifies as elementwise.flat.
//
// Planar (separate R/G/B pointers) rather than interleaved RGB on purpose: it
// keeps every access a unit-stride elementwise load, which is exactly the
// shape the elementwise recognizer claims and nothing more.
//
// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads use it; the arithmetic is float, because the Rec.601 luma weights are
// defined in full precision and a half-precision weighted sum of three channels
// would throw away mantissa the reference has no reason to lose.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rgb_to_gray(const HB_SCALAR *r, const HB_SCALAR *g,
                            const HB_SCALAR *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 0.299f * (float)r[i] + 0.587f * (float)g[i] + 0.114f * (float)b[i];
}
