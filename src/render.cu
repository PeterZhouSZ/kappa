#include <kappa/core.hpp>


__global__
void render_phong_light_kernel(
    image<float3> vm,
    image<float4> nm,
    image<rgb8> im,
    intrinsics K,
    float3 light,
    float3 view)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float3 p = vm[i];
    float3 n = make_float3(nm[i]);
    float ambient = 0.3f;
    float diffuse = 0.35f;
    float specular = 0.35f;
    float power = 20.0f;

    float3 L = normalize(light - p);
    float3 V = normalize(view - p);
    float3 R = normalize(2 * n * dot(n, L) - L);
    float intensity = ambient + diffuse * fmaxf(dot(n, L), 0.0f) +
        specular * __powf(fmaxf(dot(R, V), 0.0f), power);
    uint8_t gray = (uint8_t)(clamp(intensity, 0.0f, 1.0f) * 255.0f);
    im[i] = {gray, gray, gray};
}


__global__
void render_normal_kernel(
    image<float3> vm,
    image<float4> nm,
    image<rgb8> im,
    intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float3 n = make_float3(nm[i]);
    im[i] = {(uint8_t)((0.3f + (-n.x + 1.0f) * 0.35f) * 255.0f),
             (uint8_t)((0.3f + (-n.y + 1.0f) * 0.35f) * 255.0f),
             (uint8_t)((0.3f + (-n.z + 1.0f) * 0.35f) * 255.0f)};
}


void render_phong_light(
    const image<float3>* vm,
    const image<float4>* nm,
    image<rgb8>* im,
    intrinsics K,
    float3 light,
    float3 view)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_phong_light_kernel<<<grid_size, block_size>>>(
        vm->cuda(), nm->cuda(), im->cuda(), K, light, view);
}


void render_normal(
    const image<float3>* vm,
    const image<float4>* nm,
    image<rgb8>* im,
    intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_normal_kernel<<<grid_size, block_size>>>(
        vm->cuda(), nm->cuda(), im->cuda(), K);
}
