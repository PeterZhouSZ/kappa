#include <kappa/core.hpp>
#define ZBUFFER_SCALE 100000


__global__
void raycast_volume_kernel(volume<sdf32f_t> vol, image<float3> vm, image<float4> nm, intrinsics K, mat4x4 T, float mu, float near, float far)
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

    float ft = nearest_tsdf(vol, p);
    float ftt;
    float step = 0.8f * mu;
    float min_step = vol.voxel_size;
    for (; z <= far; z += step) {
        p = origin + direction * z;
        ftt = interp_tsdf(vol, p);
        if (ftt < 0.0f) break;
        step = fmaxf(0.8f * ftt * mu, min_step);
        ft = ftt;
    }

    if (ftt < 0.0f) z += step * ftt / (ft - ftt);
    else z = -1.0f;

    int i = u + v * K.width;
    vm.data[i] = {0.0f, 0.0f, 0.0f};
    nm.data[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (z >= 0.0f) {
        p = origin + direction * z;
        vm.data[i] = p;
        nm.data[i] = make_float4(grad_tsdf(vol, p));
    }
}


__global__
void raycast_z_buffer_kernel(cloud<surfel32f_t> pc, image<uint32_t> zbuf, intrinsics K, mat4x4 T)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= pc.size) return;

    float3 p = pc.data[k].pos;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    int i = u + v * K.width;
    uint32_t z = q.z * ZBUFFER_SCALE;
    atomicMin(&zbuf.data[i], z);
}


__global__
void raycast_index_kernel(cloud<surfel32f_t> pc, image<uint32_t> zbuf, image<uint4> im, intrinsics K, mat4x4 T)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= pc.size) return;

    float3 p = pc.data[k].pos;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    int i = u + v * K.width;
    uint32_t z = q.z * ZBUFFER_SCALE;
    if (z > zbuf.data[i]) return;
    im.data[i].x = k + 1;
}


__global__
void raycast_cloud_kernel(cloud<surfel32f_t> pc, image<uint4> im, image<float3> vm, image<float4> nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    vm.data[i] = {0.0f, 0.0f, 0.0f};
    nm.data[i] = {0.0f, 0.0f, 0.0f};

    int k = im.data[i].x;
    if (k == 0) return;

    vm.data[i] = pc.data[k - 1].pos;
    nm.data[i] = make_float4(pc.data[k - 1].normal);
}


void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vm, image<float4>* nm, intrinsics K, mat4x4 T, float mu, float near, float far)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    raycast_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), vm->gpu(), nm->gpu(), K, T, mu, near, far);
}


void raycast_cloud(const cloud<surfel32f_t>* pc, image<float3>* vm, image<float4>* nm, image<uint4>* im, intrinsics K, mat4x4 T)
{
    static image<uint32_t> zbuf;
    zbuf.allocate(K.width, K.height, DEVICE_CUDA);
    zbuf.clear(0xff);
    im->clear();
    {
        unsigned int block_size = 512;
        unsigned int grid_size = divup(pc->size, block_size);
        raycast_z_buffer_kernel<<<grid_size, block_size>>>(pc->gpu(), zbuf.gpu(), K, T.inverse());
        raycast_index_kernel<<<grid_size, block_size>>>(pc->gpu(), zbuf.gpu(), im->gpu(), K, T.inverse());
    }
    {
        dim3 block_size(16, 16);
        dim3 grid_size;
        grid_size.x = divup(K.width, block_size.x);
        grid_size.y = divup(K.height, block_size.y);
        raycast_cloud_kernel<<<grid_size, block_size>>>(pc->gpu(), im->gpu(), vm->gpu(), nm->gpu(), K);
    }
}
