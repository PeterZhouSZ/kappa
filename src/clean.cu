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
}
