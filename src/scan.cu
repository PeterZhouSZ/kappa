#include <kappa/common.hpp>


__global__
void prescan_blelloch_kernel(uint32_t* a, uint32_t* sum, uint32_t* bsum, int n)
{
}


void sum_scan_cuda(uint32_t* a, uint32_t* sum, int n)
{
    int block_size = 512;
    int elems_per_block = 2 * block_size;
    int grid_size = divup(n, elems_per_block);

    uint32_t* bsum = NULL;
    cudaMalloc((void**)&bsum, sizeof(uint32_t) * grid_size);
    cudaMemset(bsum, 0, sizeof(uint32_t) * grid_size);

    prescan_blelloch_kernel<<<grid_size, block_size, sizeof(uint32_t) * elems_per_block>>>(a, sum, bsum, n);

    cudaFree(bsum);
}
