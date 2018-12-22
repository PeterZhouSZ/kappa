#include <kappa/common.hpp>


__global__
void prescan_blelloch_kernel(uint32_t* a, uint32_t* sum, uint32_t* bsum, int n)
{
    extern __shared__ uint32_t ssum[];
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    ssum[2 * tid] = 0;
    ssum[2 * tid + 1] = 0;
    __syncthreads();

    ssum[2 * tid] = a[2 * gid];
    ssum[2 * tid + 1] = a[2 * gid + 1];
    __syncthreads();

    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ssum[bi] += ssum[ai];
        }
        offset <<= 1;
    }

    if (threadIdx.x == 0) {
        if (bsum) bsum[blockIdx.x] = ssum[2 * blockDim.x - 1];
        ssum[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d <= blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            uint32_t t = ssum[ai];
            ssum[ai] = ssum[bi];
            ssum[bi] += t;
        }
    }
    __syncthreads();

    sum[2 * gid] = ssum[2 * tid];
    sum[2 * gid + 1] = ssum[2 * tid + 1];
}


__global__
void prescan_add_block_kernel(uint32_t* a, uint32_t* sum, uint32_t* bsum, int n)
{
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = bsum[blockIdx.x];
    sum[i] += v;
    sum[i + blockDim.x] += v;
}


uint32_t prescan(uint32_t* a, uint32_t* sum, int n)
{
    static int block_size = 512;
    static int elems_per_block = 2 * block_size;
    int grid_size = divup(n, elems_per_block);

    static uint32_t* bsum = nullptr;
    if (bsum == nullptr)
        CUDA_MALLOC_T(bsum, uint32_t, elems_per_block);
    CUDA_MEMSET(bsum, 0, sizeof(uint32_t) * elems_per_block);

    prescan_blelloch_kernel<<<grid_size, block_size,
        sizeof(uint32_t) * elems_per_block>>>(a, sum, bsum, n);

    if (grid_size <= elems_per_block)
        prescan_blelloch_kernel<<<1, block_size,
            sizeof(uint32_t) * elems_per_block>>>(
                bsum, bsum, nullptr, grid_size);
    prescan_add_block_kernel<<<grid_size, block_size>>>(a, sum, bsum, n);

    uint32_t s;
    CUDA_MEMCPY_DEVICE_TO_HOST(&s, &sum[n - 1], sizeof(uint32_t));
    return s;
}
