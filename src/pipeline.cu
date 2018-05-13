#include <kinfu/pipeline.hpp>


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


static void compute_vertex_map(const image<uint16_t>* dm, image<float3>* vm, intrinsics K, float cutoff)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_vertex_kernel<<<grid_size, block_size>>>(dm->gpu(), vm->gpu(), K, cutoff);
}


pipeline::pipeline()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}


pipeline::~pipeline()
{
    for (int i = 0; i < max_pyramid_level; ++i) {
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
    vm[0].resize(dm[0].width, dm[0].height);
    compute_vertex_map(&dm[0], &vm[0], cam->K, cutoff);
}

}
