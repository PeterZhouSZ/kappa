#include <kappa/core.hpp>


__global__
void mark_stable_surfel_kernel(
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
    if (unstable && duration > period) mask[k] = 0;
    else mask[k] = 1;
}


__global__
void remove_unstable_surfel_kernel(
    cloud<surfel> input,
    cloud<surfel> output,
    uint32_t* mask,
    uint32_t* sum,
    int offset)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= output.size) return;
    if (mask[k]) output[sum[k]] = input[k];
    if (k >= offset) {
        output[k].weight = 0.0f;
        output[k].radius = 1.0f;
    }
}


void cleanup(cloud<surfel>* pcd,
             float maxw,
             int timestamp,
             int period)
{
    static uint32_t* mask = nullptr;
    static uint32_t* sum = nullptr;
    if (!mask) cudaMalloc((void**)&mask, sizeof(uint32_t) * pcd->capacity);
    if (!sum) cudaMalloc((void**)&sum, sizeof(uint32_t) * pcd->capacity);

    uint32_t block_size = 512;
    uint32_t grid_size = divup(pcd->size, block_size);
    mark_stable_surfel_kernel<<<grid_size, block_size>>>(
        pcd->cuda(), mask, maxw, timestamp, period);

    int size = prescan(mask, sum, pcd->size);
    cloud<surfel> other = pcd->clone();
    remove_unstable_surfel_kernel<<<grid_size, block_size>>>(
        other.cuda(), pcd->cuda(), mask, sum, size);
    pcd->size = size;
    other.free();
}
