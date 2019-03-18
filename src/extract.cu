#include <kappa/core.hpp>


__global__
void extract_isosurface_volume_kernel(
    volume<voxel> vol,
    array<vertex> va,
    int* size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.shape.x || y >= vol.shape.y || z >= vol.shape.z) return;

    float fx = vol(x, y, z);
    if (fabs(fx) >= 0.05f) return;

    float3 p;
    p.x = vol.offset.x + x * vol.voxel_size;
    p.y = vol.offset.y + y * vol.voxel_size;
    p.z = vol.offset.z + z * vol.voxel_size;

    int i = atomicAdd(size, 1);
    va[i].pos = p;
    va[i].normal = vol.grad(p);
    va[i].color = vol.color(p);
}


int extract_isosurface_volume(const volume<voxel> vol, array<vertex>* va)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol.shape.x, block_size.x);
    grid_size.y = divup(vol.shape.y, block_size.y);
    grid_size.z = divup(vol.shape.z, block_size.z);

    int* sum = nullptr;
    cudaMalloc((void**)&sum, sizeof(int));
    cudaMemset(sum, 0, sizeof(int));

    extract_isosurface_volume_kernel<<<grid_size, block_size>>>(
        vol.cuda(), va->cuda(), sum);

    int size;
    cudaMemcpy(&size, sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    return size;
}
