#include <kinfu/pipeline.hpp>


__global__
void bilateral_filter_kernel(image<float> dmap0, image<float> dmap1, intrinsics K, float d_sigma, float r_sigma)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    dmap1.data[i] = 0.0f;
    float p = dmap0.data[i];
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

            float q = dmap0.data[x + y * K.width];
            if (q == 0.0f) continue;

            float w_r = __expf(dx * dx * inv_r_sigma2) * __expf(dy * dy * inv_r_sigma2);
            float w_d = __expf((p - q) * (p - q) * inv_d_sigma2);
            sum += q * w_r * w_d;
            count += w_r * w_d;
        }
    }
    dmap1.data[i] = (sum / count);
}


__global__
void compute_depth_kernel(image<uint16_t> rmap, image<float> dmap, intrinsics K, float cutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float d = rmap.data[i] * 0.001f;
    if (d > cutoff) d = 0.0f;
    dmap.data[i] = d;
}


__global__
void compute_vertex_kernel(image<float> dmap, image<float3> vmap, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float d = dmap.data[i];

    vmap.data[i].x = (u - K.cx) * d / K.fx;
    vmap.data[i].y = (v - K.cy) * d / K.fy;
    vmap.data[i].z = d;
}


__global__
void compute_normal_kernel(image<float3> vmap, image<float3> nmap, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u <= 0 || u >= K.width - 1 || v <= 0 || v >= K.height - 1) return;

    float3 v00 = vmap.data[(u - 1) + v * K.width];
    float3 v10 = vmap.data[(u + 1) + v * K.width];
    float3 v01 = vmap.data[u + (v - 1) * K.width];
    float3 v11 = vmap.data[u + (v + 1) * K.width];

    float3 normal = {0.0f, 0.0f, 0.0f};
    if (v00.z != 0 && v01.z != 0 && v10.z != 0 && v11.z != 0) {
        float3 dx = v00 - v10;
        float3 dy = v01 - v11;
        normal = normalize(cross(dy, dx));
    }
    nmap.data[u + v * K.width] = normal;
}


void compute_depth_map(const image<uint16_t>* rmap, image<float>* dmap, intrinsics K, float cutoff)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_depth_kernel<<<grid_size, block_size>>>(rmap->gpu(), dmap->gpu(), K, cutoff);
}


void bilateral_filter(const image<float>* dmap0, image<float>* dmap1, intrinsics K, float d_sigma, float r_sigma)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    bilateral_filter_kernel<<<grid_size, block_size>>>(dmap0->gpu(), dmap1->gpu(), K, d_sigma, r_sigma);
}


void compute_vertex_map(const image<float>* dmap, image<float3>* vmap, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_vertex_kernel<<<grid_size, block_size>>>(dmap->gpu(), vmap->gpu(), K);
}


void compute_normal_map(const image<float3>* vmap, image<float3>* nmap, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_normal_kernel<<<grid_size, block_size>>>(vmap->gpu(), nmap->gpu(), K);
}
