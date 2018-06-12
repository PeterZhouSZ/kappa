#include <kappa/core.hpp>
#define INVALID_SURFEL 0xffffffff


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

    float sigma_r = 0.6f;
    float max_rad_dist = sqrtf(K.width * K.width * 0.25f + K.height * K.height * 0.25f);
    float inv_r_sigma2 = -1.0 / (2.0f * sigma_r * sigma_r);
    float2 uv = {(float)(u - K.cx), (float)(v - K.cy)};
    float rad_dist = length(uv) / max_rad_dist;

    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    float ftt = fminf(1.0f, dist / mu);
    float wtt = __expf(rad_dist * rad_dist * inv_r_sigma2);
    float ft  = vol.data[i].tsdf;
    float wt  = vol.data[i].weight;
    vol.data[i].tsdf = (ft * wt + ftt * wtt) / (wt + wtt);
    vol.data[i].weight = fminf(wt + wtt, max_weight);
}


__global__
void match_surfel_kernel(image<float3> vm, image<uint4> im, image<uint32_t> mm, intrinsics K, mat4x4 T)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    int k = im.data[i].x;
    mm.data[i] = 0;
    if (vm.data[i].z > 0.0f && k == 0) mm.data[i] = 1;
}


__global__
void update_index_kernel(image<uint4> im, image<uint32_t> mm, image<uint32_t> sm, intrinsics K, int base)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;
    int i = u + v * K.width;
    im.data[i].x = base + mm.data[i] * sm.data[i];
}


__global__
void integrate_cloud_kernel(cloud<surfel32f_t> pc, image<float3> vm, image<float3> nm, image<uint4> im, intrinsics K, mat4x4 T)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    int k = im.data[i].x;
    if (vm.data[i].z == 0.0f) return;

    float3 vt = pc.data[k].pos;
    float3 nt = pc.data[k].normal;
    float  wt = pc.data[k].weight;
    float3 vtt = T * vm.data[i];
    float3 ntt = rotate(T, nm.data[i]);
    float  wtt = 1.0f;

    pc.data[k].pos    = (vt * wt + vtt * wtt) / (wt + wtt);
    pc.data[k].normal = (nt * wt + ntt * wtt) / (wt + wtt);
    pc.data[k].weight = wt + wtt;
}


void integrate_volume(volume<sdf32f_t>* vol, image<float>* dm, intrinsics K, mat4x4 T, float mu, float maxw)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->dimension.x, block_size.x);
    grid_size.y = divup(vol->dimension.y, block_size.y);
    grid_size.z = divup(vol->dimension.z, block_size.z);
    integrate_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), dm->gpu(), K, T, mu, maxw);
}


void integrate_cloud(cloud<surfel32f_t>* pc, image<float3>* vm, image<float3>* nm, image<uint4>* im, intrinsics K, mat4x4 T)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);

    image<uint32_t> mm, sm;
    mm.allocate(K.width, K.height, DEVICE_CUDA);
    sm.allocate(K.width, K.height, DEVICE_CUDA);

    match_surfel_kernel<<<grid_size, block_size>>>(vm->gpu(), im->gpu(), mm.gpu(), K, T);
    int sum = sum_scan_cuda(mm.data, sm.data, K.width * K.height);
    update_index_kernel<<<grid_size, block_size>>>(im->gpu(), mm.gpu(), sm.gpu(), K, pc->size);

    integrate_cloud_kernel<<<grid_size, block_size>>>(pc->gpu(), vm->gpu(), nm->gpu(), im->gpu(), K, T);
    pc->size += sum;

    sm.deallocate();
    mm.deallocate();
}
