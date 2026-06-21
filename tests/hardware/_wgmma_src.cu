#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Warp Group MMA smoke test: m64n8k16 BF16 → F32
// All-1.0 inputs; expected output: D[0] = 16.0 (k=16 products of 1.0*1.0)
// Must launch with 128 threads (4 warps = 1 warp group)
extern "C" __global__ void wgmma_test(float *D) {
    __shared__ __nv_bfloat16 B[16][8];  // B: 16×8 BF16 in SMEM
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        B[i / 8][i % 8] = __float2bfloat16(1.0f);
    __syncthreads();

    // Encode SMEM address as wgmma descriptor (simplified form)
    // bits [13:0]  = smem_addr >> 4
    // bits [49:32] = leading dimension (stride in bytes)
    unsigned long long smem_addr = __cvta_generic_to_shared(B);
    unsigned long long b_desc = (smem_addr >> 4)
                               | ((unsigned long long)(16 * sizeof(__nv_bfloat16)) << 32);

    // A: 4 registers of BF16 pairs, all = 1.0 (BF16 0x3f80 = 1.0)
    float d0 = 0.0f;
    unsigned ra0 = 0x3f803f80u, ra1 = 0x3f803f80u,
             ra2 = 0x3f803f80u, ra3 = 0x3f803f80u;

    asm volatile("fence.proxy.async.shared::cta;\n\t");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0}, {%1,%2,%3,%4}, %5, 1, 1, 0;\n\t"
        : "+f"(d0)
        : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "l"(b_desc)
    );
    asm volatile("wgmma.commit_group.sync.aligned;\n\t");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n\t");

    if (threadIdx.x == 0) D[0] = d0;
}
