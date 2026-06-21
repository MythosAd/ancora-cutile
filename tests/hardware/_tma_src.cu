#include <cuda_runtime.h>

// TMA bulk async copy: cp.async.bulk.shared::cta.global (sm_90+)
// Copies 32 floats (128 bytes) from global in[] to shared memory,
// synchronises with mbarrier, then writes smem[0] to out[0].
extern "C" __global__ void tma_bulk(float *out, const float *in) {
    __shared__ float smem[32];
    __shared__ unsigned long long mbar;

    unsigned sp = __cvta_generic_to_shared(smem);
    unsigned mp = __cvta_generic_to_shared(&mbar);

    if (threadIdx.x == 0) {
        // mbarrier::complete_tx::bytes 模式：init 参数是期待完成的字节数，不是事务数
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;\n\t" :: "r"(mp));
        // 先 expect_tx 声明本次要传多少字节，再发出 bulk copy
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 [%0], 128;\n\t" :: "r"(mp));
        asm volatile(
            "cp.async.bulk.shared::cta.global"
            ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n\t"
            :: "r"(sp), "l"((unsigned long long)in), "r"(128u), "r"(mp));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // 等待 phase 0 完成（barrier 完成后 phase 变为 1，试等 phase=1）
        asm volatile(
            "{ .reg .pred p;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 1;\n\t"
            "@!p bra WAIT; }\n\t"
            :: "r"(mp));
        out[0] = smem[0];
    }
}
