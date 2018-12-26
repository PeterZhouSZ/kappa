#include <kappa/core.hpp>


__global__
void raycast_volume_kernel(
    volume<voxel> vol,
    image<float3> vm,
    image<float4> nm,
    intrinsics K,
    mat4x4 T,
    float mu,
    float near,
    float far)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    float3 q;
    q.x = (u - K.cx) / K.fx;
    q.y = (v - K.cy) / K.fy;
    q.z = 1.0f;

    float3 origin = {T.m03, T.m13, T.m23};
    float3 direction = rotate(T, q);

    float z = near;
    float3 p = origin + direction * z;

    float ft = vol.nearest(p);
    float ftt;
    float step = 0.8f * mu;
    float min_step = vol.voxel_size;
    for (; z <= far; z += step) {
        p = origin + direction * z;
        ftt = vol.interp(p);
        if (ftt < 0.0f) break;
        step = fmaxf(0.8f * ftt * mu, min_step);
        ft = ftt;
    }

    if (ftt < 0.0f) z += step * ftt / (ft - ftt);
    else z = -1.0f;

    int i = u + v * K.width;
    vm[i] = {0.0f, 0.0f, 0.0f};
    nm[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (z >= 0.0f) {
        p = origin + direction * z;
        vm[i] = p;
        nm[i] = make_float4(vol.grad(p));
    }
}


__global__
void raycast_z_buffer_kernel(
    cloud<surfel> pcd,
    image<uint32_t> zbuf,
    intrinsics K,
    mat4x4 T)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= pcd.size) return;

    float3 p = pcd[k].pos;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float f = 0.5 * (K.fx + K.fy);
    int r = pcd[k].radius * f / q.z + 0.5f;
    int r2 = r * r; // radius squared

    uint32_t z = q.z * ZBUFFER_SCALE;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int x = u + dx;
            int y = v + dy;
            if (x < 0 || x >= K.width || y < 0 || y >= K.height) continue;
            if (dx * dx + dy * dy > r2) continue;

            int i = x + y * K.width;
            atomicMin(&zbuf[i], z);
        }
    }
}


__global__
void raycast_index_kernel(
    cloud<surfel> pcd,
    image<uint32_t> zbuf,
    image<uint32_t> idm,
    intrinsics K,
    mat4x4 T)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= pcd.size) return;

    float3 p = pcd[k].pos;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float f = 0.5 * (K.fx + K.fy);
    int r = pcd[k].radius * f / q.z + 0.5f;
    int r2 = r * r; // radius squared

    uint32_t z = q.z * ZBUFFER_SCALE;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int x = u + dx;
            int y = v + dy;
            if (x < 0 || x >= K.width || y < 0 || y >= K.height) continue;
            if (dx * dx + dy * dy > r2) continue;

            int i = x + y * K.width;
            if (z > zbuf[i]) continue;
            idm[i] = k + 1;
        }
    }
}


__global__
void raycast_cloud_kernel(
    cloud<surfel> pcd,
    image<uint32_t> idm,
    image<float3> vm,
    image<float4> nm,
    intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    vm[i] = {0.0f, 0.0f, 0.0f};
    nm[i] = {0.0f, 0.0f, 0.0f};

    int k = idm[i];
    if (k == 0) return;

    vm[i] = pcd[k - 1].pos;
    nm[i] = make_float4(pcd[k - 1].normal);
}


void raycast(const volume<voxel> vol,
             image<float3>* vm,
             image<float4>* nm,
             intrinsics K,
             mat4x4 T,
             float mu,
             float near,
             float far)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    raycast_volume_kernel<<<grid_size, block_size>>>(
        vol.cuda(), vm->cuda(), nm->cuda(), K, T, mu, near, far);
}


void raycast(const cloud<surfel> pcd,
             image<float3>* vm,
             image<float4>* nm,
             image<uint32_t>* idm,
             intrinsics K,
             mat4x4 T)
{
    static image<uint32_t> zbuf;
    zbuf.resize(K.width, K.height, DEVICE_CUDA);
    zbuf.clear(0xff);
    idm->clear();
    {
        unsigned int block_size = 512;
        unsigned int grid_size = divup(pcd.size, block_size);
        raycast_z_buffer_kernel<<<grid_size, block_size>>>(
            pcd.cuda(), zbuf.cuda(), K, T.inverse());
        raycast_index_kernel<<<grid_size, block_size>>>(
            pcd.cuda(), zbuf.cuda(), idm->cuda(), K, T.inverse());
    }
    {
        dim3 block_size(16, 16);
        dim3 grid_size;
        grid_size.x = divup(K.width, block_size.x);
        grid_size.y = divup(K.height, block_size.y);
        raycast_cloud_kernel<<<grid_size, block_size>>>(
            pcd.cuda(), idm->cuda(), vm->cuda(), nm->cuda(), K);
    }
}
