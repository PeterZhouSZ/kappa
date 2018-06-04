#include <kinfu/pipeline.hpp>


__global__
void integrate_volume_kernel(volume<sdf32f_t> vol, image<float> dm, intrinsics K, mat4x4 T, float mu, float max_weight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.dimension.x || y >= vol.dimension.y || z >= vol.dimension.z) return;

    float3 p ={(float)x, (float)y, (float)z};
    p = vol.offset + p * vol.voxel_size;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float d = dm.data[u + v * K.width];
    if (d == 0.0f) return;

    float dist = d - q.z;
    if (dist <= -mu) return;

    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    float ftt = fminf(1.0f, dist / mu);
    float wtt = 1.0f;
    float ft  = vol.data[i].tsdf;
    float wt  = vol.data[i].weight;
    vol.data[i].tsdf = (ft * wt + ftt * wtt) / (wt + wtt);
    vol.data[i].weight = fminf(wt + wtt, max_weight);
}


void integrate_volume(const volume<sdf32f_t>* vol, image<float>* dm, intrinsics K, mat4x4 T, float mu, float max_weight)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->dimension.x, block_size.x);
    grid_size.y = divup(vol->dimension.y, block_size.y);
    grid_size.z = divup(vol->dimension.z, block_size.z);
    integrate_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), dm->gpu(), K, T, mu, max_weight);
}
