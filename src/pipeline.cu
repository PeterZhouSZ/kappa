#include <kinfu/pipeline.hpp>
#include <kinfu/math.hpp>


namespace kinfu {

__global__
void compute_vertex_kernel(uint16_t* dm, float3* vm, intrinsics K, float cutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float d = dm[i] * 0.001f;
    if (d > cutoff) d = 0.0f;

    vm[i].x = (u - K.cx) * d / K.fx;
    vm[i].y = (v - K.cy) * d / K.fy;
    vm[i].z = d;
}


__global__
void compute_normal_kernel(float3* vm, float3* nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u <= 0 || u >= K.width - 1 || v <= 0 || v >= K.height - 1) return;

    float3 v00 = vm[(u - 1) + v * K.width];
    float3 v10 = vm[(u + 1) + v * K.width];
    float3 v01 = vm[u + (v - 1) * K.width];
    float3 v11 = vm[u + (v + 1) * K.width];

    float3 normal = {0.0f, 0.0f, 0.0f};
    if (v00.z != 0 && v01.z != 0 && v10.z != 0 && v11.z != 0) {
        float3 dx = v00 - v10;
        float3 dy = v01 - v11;
        normal = normalize(cross(dy, dx));
    }
    nm[u + v * K.width] = normal;
}


static void compute_vertex_map(const image<uint16_t>* dm, image<float3>* vm, intrinsics K, float cutoff)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_vertex_kernel<<<grid_size, block_size>>>(dm->gpu(), vm->gpu(), K, cutoff);
}


static void compute_normal_map(const image<float3>* vm, image<float3>* nm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_normal_kernel<<<grid_size, block_size>>>(vm->gpu(), nm->gpu(), K);
}


pipeline::pipeline()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}


pipeline::~pipeline()
{
    for (int i = 0; i < num_levels; ++i) {
        dm[i].deallocate();
        cm[i].deallocate();
    }
}


void pipeline::register_camera(camera* cam)
{
    this->cam = cam;
}


void pipeline::process()
{
    cam->read(&dm[0], &cm[0]);
    preprocess();
}


void pipeline::preprocess()
{
    for (int i = 0; i < num_levels; ++i) {
        vm[i].resize(dm[i].width, dm[i].height, ALLOCATOR_DEVICE);
        nm[i].resize(dm[i].width, dm[i].height, ALLOCATOR_DEVICE);
        compute_vertex_map(&dm[i], &vm[i], cam->K, cutoff);
        compute_normal_map(&vm[i], &nm[i], cam->K);
    }
}

}
