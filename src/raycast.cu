#include <kinfu/pipeline.hpp>


__global__
void raycast_volume_kernel(volume<sdf32f_t> vol, image<float3> vm, image<float3> nm, intrinsics K, mat4x4 T, float near, float far)
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
    float step = vol.voxel_size;
    for (; z <= far; z += step) {
        p = origin + direction * z;
        ftt = interp_tsdf(vol, p);
        if (ftt < 0.0f) break;
        ft = ftt;
    }

    if (ftt < 0.0f) z += step * ftt / (ft - ftt);
    else z = -1.0f;

    int i = u + v * K.width;
    vm.data[i] = {0.0f, 0.0f, 0.0f};
    nm.data[i] = {0.0f, 0.0f, 0.0f};
    if (z >= 0.0f) {
        p = origin + direction * z;
        vm.data[i] = p;
        nm.data[i] = grad_tsdf(vol, p);
    }
}


void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vm, image<float3>* nm, intrinsics K, mat4x4 T, float near, float far)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    raycast_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), vm->gpu(), nm->gpu(), K, T, near, far);
}
