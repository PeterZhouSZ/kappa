#include <kappa/core.hpp>


__global__
void mark_unstable_surfel_kernel(
    cloud<surfel> pcd,
    uint32_t* mask,
    float maxw,
    int timestamp,
    int period)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= pcd.size) return;
    bool unstable = (pcd[k].weight < maxw);
    int duration = timestamp - pcd[k].timestamp;
    if (unstable && duration > period)
        mask[k] = 1;
}


void cleanup(cloud<surfel>* pcd,
             float maxw,
             int timestamp,
             int period)
{
    static uint32_t* mask = nullptr;
    static uint32_t* sum = nullptr;
    if (mask == nullptr) CUDA_MALLOC_T(mask, uint32_t, pcd->capacity);
    if (sum == nullptr) CUDA_MALLOC_T(sum, uint32_t, pcd->capacity);
    CUDA_MEMSET(mask, 0, sizeof(uint32_t) * pcd->capacity);

    unsigned int block_size = 512;
    unsigned int grid_size = divup(pcd->size, block_size);
    mark_unstable_surfel_kernel<<<grid_size, block_size>>>(
        pcd->cuda(), mask, maxw, timestamp, period);

    int size = prescan(mask, sum, pcd->size);
    printf("%d\n", size);
}
