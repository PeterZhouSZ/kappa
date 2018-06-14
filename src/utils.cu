#include <kappa/core.hpp>


__global__
void depth_bilateral_kernel(image<float> dm0, image<float> dm1, intrinsics K, float d_sigma, float r_sigma)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    dm1.data[i] = 0.0f;
    float p = dm0.data[i];
    if (p == 0.0f) return;

    float sum = 0.0f;
    float count = 0.0f;
    float inv_r_sigma2 = -1.0f / (2.0f * r_sigma * r_sigma);
    float inv_d_sigma2 = -1.0f / (2.0f * d_sigma * d_sigma);

    int radius = 2;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int x = u + dx;
            int y = v + dy;
            if (x < 0 || x >= K.width || y < 0 || y >= K.height) continue;

            float q = dm0.data[x + y * K.width];
            if (q == 0.0f) continue;

            float w_r = __expf(dx * dx * inv_r_sigma2) * __expf(dy * dy * inv_r_sigma2);
            float w_d = __expf((p - q) * (p - q) * inv_d_sigma2);
            sum += q * w_r * w_d;
            count += w_r * w_d;
        }
    }
    dm1.data[i] = (sum / count);
}


__global__
void compute_depth_kernel(image<uint16_t> rm, image<float> dm, intrinsics K, float cutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float d = rm.data[i] * 0.001f;
    if (d > cutoff) d = 0.0f;
    dm.data[i] = d;
}


__global__
void compute_vertex_kernel(image<float> dm, image<float3> vm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float d = dm.data[i];

    vm.data[i].x = (u - K.cx) * d / K.fx;
    vm.data[i].y = (v - K.cy) * d / K.fy;
    vm.data[i].z = d;
}


__global__
void compute_normal_kernel(image<float3> vm, image<float3> nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u <= 0 || u >= K.width - 1 || v <= 0 || v >= K.height - 1) return;

    float3 v00 = vm.data[(u - 1) + v * K.width];
    float3 v10 = vm.data[(u + 1) + v * K.width];
    float3 v01 = vm.data[u + (v - 1) * K.width];
    float3 v11 = vm.data[u + (v + 1) * K.width];

    float3 normal = {0.0f, 0.0f, 0.0f};
    if (v00.z != 0 && v01.z != 0 && v10.z != 0 && v11.z != 0) {
        float3 dx = v00 - v10;
        float3 dy = v01 - v11;
        normal = normalize(cross(dy, dx));
    }
    nm.data[u + v * K.width] = normal;
}


__global__
void compute_normal_radius_kernel(image<float3> vm, image<float4> nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u <= 0 || u >= K.width - 1 || v <= 0 || v >= K.height - 1) return;

    int i = u + v * K.width;
    float3 v00 = vm.data[(u - 1) + v * K.width];
    float3 v10 = vm.data[(u + 1) + v * K.width];
    float3 v01 = vm.data[u + (v - 1) * K.width];
    float3 v11 = vm.data[u + (v + 1) * K.width];

    float3 normal = {0.0f, 0.0f, 0.0f};
    if (v00.z != 0 && v01.z != 0 && v10.z != 0 && v11.z != 0) {
        float3 dx = v00 - v10;
        float3 dy = v01 - v11;
        normal = normalize(cross(dy, dx));
    }

    float r = 0.0f;
    if (length(normal) > 0.0f) {
        float d = vm.data[i].z;
        float f = 0.5f * (K.fx + K.fy);
        r = sqrtf(2.0f) * d / f;
    }

    nm.data[i].x = normal.x;
    nm.data[i].y = normal.y;
    nm.data[i].z = normal.z;
    nm.data[i].w = r;
}


__global__
void reset_volume_kernel(volume<sdf32f_t> vol)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.dimension.x || y >= vol.dimension.y || z >= vol.dimension.z) return;
    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    vol.data[i].tsdf = 1.0f;
    vol.data[i].weight = 0.0f;
}


void compute_depth_map(const image<uint16_t>* rm, image<float>* dm, intrinsics K, float cutoff)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_depth_kernel<<<grid_size, block_size>>>(rm->gpu(), dm->gpu(), K, cutoff);
}


void depth_bilateral(const image<float>* dm0, image<float>* dm1, intrinsics K, float d_sigma, float r_sigma)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    depth_bilateral_kernel<<<grid_size, block_size>>>(dm0->gpu(), dm1->gpu(), K, d_sigma, r_sigma);
}


void compute_vertex_map(const image<float>* dm, image<float3>* vm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_vertex_kernel<<<grid_size, block_size>>>(dm->gpu(), vm->gpu(), K);
}


void compute_normal_map(const image<float3>* vm, image<float3>* nm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_normal_kernel<<<grid_size, block_size>>>(vm->gpu(), nm->gpu(), K);
}


void compute_normal_radius_map(const image<float3>* vm, image<float4>* nm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_normal_radius_kernel<<<grid_size, block_size>>>(vm->gpu(), nm->gpu(), K);
}


void reset_volume(volume<sdf32f_t>* vol)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->dimension.x, block_size.x);
    grid_size.y = divup(vol->dimension.y, block_size.y);
    grid_size.z = divup(vol->dimension.z, block_size.z);
    reset_volume_kernel<<<grid_size, block_size>>>(vol->gpu());
}
