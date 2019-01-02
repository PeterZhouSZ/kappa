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
    float3 ambient = {0.2f, 0.2f, 0.2f};
    float3 diffuse = {1.0f, 0.706f, 0.0f};
    float3 specular = {0.2f, 0.2f, 0.2f};
    float shininess = 20.0f;

    float3 L = normalize(light - p);
    float3 V = normalize(view - p);
    float3 H = normalize(L + V);
    float3 color = ambient + diffuse * fmaxf(dot(n, L), 0.0f) +
        specular * __powf(fmaxf(dot(n, H), 0.0f), shininess);
    color = clamp(color, 0.0f, 1.0f) * 255.0f;
    im[i] = {(uint8_t)color.x, (uint8_t)color.y, (uint8_t)color.z};
}


__global__
void render_phong_light_kernel(
    image<float3> vm,
    image<float4> nm,
    image<float3> cm,
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
    float3 ambient = {0.2f, 0.2f, 0.2f};
    float3 diffuse = cm[i];
    float3 specular = {0.2f, 0.2f, 0.2f};
    float shininess = 20.0f;

    float3 L = normalize(light - p);
    float3 V = normalize(view - p);
    float3 H = normalize(L + V);
    float3 color = ambient + diffuse * fmaxf(dot(n, L), 0.0f) +
        specular * __powf(fmaxf(dot(n, H), 0.0f), shininess);
    color = clamp(color, 0.0f, 1.0f) * 255.0f;
    im[i] = {(uint8_t)color.x, (uint8_t)color.y, (uint8_t)color.z};
}


void render_phong_light(
    const image<float3> vm,
    const image<float4> nm,
    image<rgb8>* im,
    intrinsics K,
    float3 light,
    float3 view)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width,  block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_phong_light_kernel<<<grid_size, block_size>>>(
        vm.cuda(), nm.cuda(), im->cuda(), K, light, view);
}


void render_phong_light(
    const image<float3> vm,
    const image<float4> nm,
    const image<float3> cm,
    image<rgb8>* im,
    intrinsics K,
    float3 light,
    float3 view)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width,  block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_phong_light_kernel<<<grid_size, block_size>>>(
        vm.cuda(), nm.cuda(), cm.cuda(), im->cuda(), K, light, view);
}
