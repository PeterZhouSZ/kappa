#include <kappa/core.hpp>


__global__
void integrate_volume_kernel(
    volume<voxel> vol,
    image<float> dm,
    image<float3> cm,
    intrinsics K,
    mat4x4 T,
    float mu,
    float maxw)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.shape.x || y >= vol.shape.y || z >= vol.shape.z) return;

    float3 p = {(float)x, (float)y, (float)z};
    p = vol.offset + p * vol.voxel_size;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float d = dm[u + v * K.width];
    if (d == 0.0f) return;

    float dist = d - q.z;
    if (dist <= -mu) return;

    float sigma_r = 0.6f;
    float max_rad_dist = sqrtf(
        K.width * K.width * 0.25f +
        K.height * K.height * 0.25f);
    float inv_r_sigma2 = -1.0 / (2.0f * sigma_r * sigma_r);
    float2 uv = {(float)(u - K.cx), (float)(v - K.cy)};
    float rad_dist = length(uv) / max_rad_dist;

    int i = x + y * vol.shape.x + z * vol.shape.x * vol.shape.y;
    float  ftt = fminf(1.0f, dist / mu);
    float  wtt = __expf(rad_dist * rad_dist * inv_r_sigma2);
    float3 ctt = cm[u + v * K.width];
    float  ft  = vol[i].tsdf;
    float  wt  = vol[i].weight;
    float3 ct  = vol[i].color;

    vol[i].tsdf   = (ft * wt + ftt * wtt) / (wt + wtt);
    vol[i].color  = (ct * wt + ctt * wtt) / (wt + wtt);
    vol[i].weight = fminf(wt + wtt, maxw);
    vol[i].color  = clamp(vol[i].color, 0.0f, 1.0f);
}


__global__
void match_surfel_kernel(
    cloud<surfel> pcd,
    image<float3> vm,
    image<uint32_t> idm,
    image<uint32_t> mm,
    intrinsics K,
    mat4x4 T)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    mm[i] = 0;
    if (vm[i].z == 0.0f) return;

    int k = idm[i] - 1;
    if (k >= 0) {
        float3 vtt = T * vm[i];
        float3 vt = pcd[k].pos;
        if (fabs(vt.z - vtt.z) < 0.01f) return;
    }
    mm[i] = 1;
}


__global__
void update_index_kernel(
    image<uint32_t> idm,
    image<uint32_t> mm,
    image<uint32_t> sm,
    intrinsics K,
    int offset)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;
    int i = u + v * K.width;
    idm[i] = mm[i] ? offset + sm[i] + 1 : idm[i];
}


__global__
void integrate_cloud_kernel(
    cloud<surfel> pcd,
    image<float3> vm,
    image<float4> nm,
    image<float3> cm,
    image<uint32_t> idm,
    intrinsics K,
    mat4x4 T,
    int timestamp,
    float delta_r)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    int k = idm[i] - 1;
    if (vm[i].z == 0.0f || k < 0) return;

    float sigma_r = 0.6f;
    float max_rad_dist = sqrtf(
        K.width * K.width * 0.25f +
        K.height * K.height * 0.25f);
    float inv_r_sigma2 = -1.0 / (2.0f * sigma_r * sigma_r);
    float2 uv = {(float)(u - K.cx), (float)(v - K.cy)};
    float rad_dist = length(uv) / max_rad_dist;

    float3 normal = make_float3(nm[i]);
    float3 vtt = T * vm[i];
    float3 ntt = rotate(T, normal);
    float3 ctt = cm[i];
    float  rtt = nm[i].w;
    float  wtt = __expf(rad_dist * rad_dist * inv_r_sigma2);
    float3 vt = pcd[k].pos;
    float3 nt = pcd[k].normal;
    float3 ct = pcd[k].color;
    float  rt = pcd[k].radius;
    float  wt = pcd[k].weight;

    pcd[k].weight = wt + wtt;
    pcd[k].timestamp = timestamp;
    if (rtt > rt * delta_r) return;
    pcd[k].pos    = (vt * wt + vtt * wtt) / pcd[k].weight;
    pcd[k].normal = (nt * wt + ntt * wtt) / pcd[k].weight;
    pcd[k].color  = (ct * wt + ctt * wtt) / pcd[k].weight;
    pcd[k].radius = (rt * wt + rtt * wtt) / pcd[k].weight;
    pcd[k].normal = normalize(pcd[k].normal);
    pcd[k].color  = clamp(pcd[k].color, 0.0f, 1.0f);
}


void integrate_volume(
    volume<voxel>* vol,
    const image<float> dm,
    const image<float3> cm,
    intrinsics K,
    mat4x4 T,
    float mu,
    float maxw)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->shape.x, block_size.x);
    grid_size.y = divup(vol->shape.y, block_size.y);
    grid_size.z = divup(vol->shape.z, block_size.z);
    integrate_volume_kernel<<<grid_size, block_size>>>(
        vol->cuda(), dm.cuda(), cm.cuda(), K, T.inverse(), mu, maxw);
}


void integrate_cloud(
    cloud<surfel>* pcd,
    const image<float3> vm,
    const image<float4> nm,
    const image<float3> cm,
    const image<uint32_t> idm,
    intrinsics K,
    mat4x4 T,
    int timestamp,
    float delta_r)
{
    static image<uint32_t> mm, sm;
    mm.resize(K.width, K.height, DEVICE_CUDA);
    sm.resize(K.width, K.height, DEVICE_CUDA);

    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width,  block_size.x);
    grid_size.y = divup(K.height, block_size.y);

    match_surfel_kernel<<<grid_size, block_size>>>(
        pcd->cuda(), vm.cuda(), idm.cuda(), mm.cuda(), K, T);

    int sum = prescan(mm.data, sm.data, K.width * K.height);
    update_index_kernel<<<grid_size, block_size>>>(
        idm.cuda(), mm.cuda(), sm.cuda(), K, pcd->size);
    integrate_cloud_kernel<<<grid_size, block_size>>>(
        pcd->cuda(), vm.cuda(), nm.cuda(), cm.cuda(),
        idm.cuda(), K, T, timestamp, delta_r);
    pcd->size += sum;
}
