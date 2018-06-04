#include <kinfu/pipeline.hpp>


__global__
void render_phong_light_kernel(image<rgb8_t> im, image<float3> vm, image<float3> nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float3 light = {0.0f, 0.0f, 0.0f};
    float3 view = {0.0f, 0.0f, 0.0f};
    float3 p = vm.data[i];
    float3 n = nm.data[i];
    float ambient = 0.1f;
    float diffuse = 0.5f;
    float specular = 0.2f;
    float power = 20.0f;

    float3 L = normalize(light - p);
    float3 V = normalize(view - p);
    float3 R = normalize(2 * n * dot(n, L) - L);
    float intensity = ambient + diffuse * fmaxf(dot(n, L), 0.0f) + specular * __powf(fmaxf(dot(R, V), 0.0f), power);
    uint8_t gray = (uint8_t)(clamp(intensity, 0.0f, 1.0f) * 255.0f);
    im.data[i] = {gray, gray, gray};
}

__global__
void render_normal_kernel(image<rgb8_t> im, image<float3> nm, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float3 color = 255.0f * fabs(nm.data[i]);
    im.data[i] = {(uint8_t)color.x, (uint8_t)color.y, (uint8_t)color.z};
}


void render_phong_light(image<rgb8_t>* im, const image<float3>* vm, const image<float3>* nm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_phong_light_kernel<<<grid_size, block_size>>>(im->gpu(), vm->gpu(), nm->gpu(), K);
}


void render_normal(image<rgb8_t>* im, const image<float3>* nm, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_normal_kernel<<<grid_size, block_size>>>(im->gpu(), nm->gpu(), K);
}
