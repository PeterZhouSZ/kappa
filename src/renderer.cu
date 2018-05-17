#include <kinfu/renderer.hpp>
#include <kinfu/math.hpp>


__global__
void render_phong_light_kernel(image<rgb8_t> im, image<float3> vmap, image<float3> nmap, intrinsics K)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    float3 light = {0.0f, 0.0f, 0.0f};
    float3 view = {0.0f, 0.0f, 0.0f};
    float3 p = vmap.data[i];
    float3 n = nmap.data[i];
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


static void render_phong_light(image<rgb8_t>* im, const image<float3>* vmap, const image<float3>* nmap, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    render_phong_light_kernel<<<grid_size, block_size>>>(im->gpu(), vmap->gpu(), nmap->gpu(), K);
}


void renderer::render_phong(image<rgb8_t>* im, const image<float3>* vmap, const image<float3>* nmap)
{
    im->resize(K.width, K.height, ALLOCATOR_MAPPED);
    render_phong_light(im, vmap, nmap, K);
}
