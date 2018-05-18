#include <kinfu/pipeline.hpp>
#include <float.h>
#include <kinfu/math.hpp>


__device__
float tsdf_at(volume<sdf32f_t> vol, int x, int y, int z)
{
    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    if (x < 0 || x >= vol.dimension.x ||
        y < 0 || y >= vol.dimension.y ||
        z < 0 || z >= vol.dimension.z)
        return 1.0f; // cannot interpolate
    return vol.data[i].tsdf;
}


__device__
float nearest_tsdf(volume<sdf32f_t> vol, float3 p)
{
    int x = roundf((p.x - vol.offset.x) / vol.voxel_size);
    int y = roundf((p.y - vol.offset.y) / vol.voxel_size);
    int z = roundf((p.z - vol.offset.z) / vol.voxel_size);
    return tsdf_at(vol, x, y, z);
}


__device__
float interp_tsdf(volume<sdf32f_t> vol, float3 p)
{
    float3 q = (p - vol.offset) / vol.voxel_size;
    int x = (int)q.x;
    int y = (int)q.y;
    int z = (int)q.z;
    float a = q.x - x;
    float b = q.y - y;
    float c = q.z - z;

    float tsdf = 0.0f;
    tsdf += tsdf_at(vol, x + 0, y + 0, z + 0) * (1 - a) * (1 - b) * (1 - c);
    tsdf += tsdf_at(vol, x + 0, y + 0, z + 1) * (1 - a) * (1 - b) * (    c);
    tsdf += tsdf_at(vol, x + 0, y + 1, z + 0) * (1 - a) * (    b) * (1 - c);
    tsdf += tsdf_at(vol, x + 0, y + 1, z + 1) * (1 - a) * (    b) * (    c);
    tsdf += tsdf_at(vol, x + 1, y + 0, z + 0) * (    a) * (1 - b) * (1 - c);
    tsdf += tsdf_at(vol, x + 1, y + 0, z + 1) * (    a) * (1 - b) * (    c);
    tsdf += tsdf_at(vol, x + 1, y + 1, z + 0) * (    a) * (    b) * (1 - c);
    tsdf += tsdf_at(vol, x + 1, y + 1, z + 1) * (    a) * (    b) * (    c);
    return tsdf;
}


__device__
float3 grad_tsdf(volume<sdf32f_t> vol, float3 p)
{
    int x = roundf((p.x - vol.offset.x) / vol.voxel_size);
    int y = roundf((p.y - vol.offset.y) / vol.voxel_size);
    int z = roundf((p.z - vol.offset.z) / vol.voxel_size);

    float3 grad;
    float f0, f1;
    f0 = tsdf_at(vol, x - 1, y, z);
    f1 = tsdf_at(vol, x + 1, y, z);
    grad.x = (f1 - f0) / vol.voxel_size;
    f0 = tsdf_at(vol, x, y - 1, z);
    f1 = tsdf_at(vol, x, y + 1, z);
    grad.y = (f1 - f0) / vol.voxel_size;
    f0 = tsdf_at(vol, x, y, z - 1);
    f1 = tsdf_at(vol, x, y, z + 1);
    grad.z = (f1 - f0) / vol.voxel_size;
    if (length(grad) == 0.0f) return {0.0f, 0.0f, 0.0f};
    return normalize(grad);
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


__global__
void integrate_volume_kernel(volume<sdf32f_t> vol, image<float> dmap, intrinsics K, mat4x4 P, float mu)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.dimension.x || y >= vol.dimension.y || z >= vol.dimension.z) return;

    float3 p = {(float)x, (float)y, (float)z};
    float3 q = vol.offset + p * vol.voxel_size;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float d = dmap.data[u + v * K.width];
    if (d <= 0.0f) return;

    float dist = d - q.z;
    if (dist <= -mu) return;

    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    float ftt = fminf(1.0f, dist / mu);
    float wtt = 1.0f;
    float ft  = vol.data[i].tsdf;
    float wt  = vol.data[i].weight;
    vol.data[i].tsdf = (ft * wt + ftt * wtt) / (wt + wtt);
    vol.data[i].weight = wt + wtt;
}


__global__
void raycast_volume_kernel(volume<sdf32f_t> vol, image<float3> vmap, image<float3> nmap, intrinsics K, mat4x4 P, float near, float far)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    float3 q;
    q.x = (u - K.cx) / K.fx;
    q.y = (v - K.cy) / K.fy;
    q.z = 1.0f;

    float3 origin = {0.0f, 0.0f, 0.0f};
    float3 direction = q;

    float z = near;
    float3 p = origin + direction * z;

    float ft, ftt;
    float step = vol.voxel_size;
    for (; z <= far; z += step) {
        p = origin + direction * z;
        ftt = interp_tsdf(vol, p);
        if (ftt < 0.0f) break;
        ft = ftt;
    }

    if (ftt < 0.0f) z += step * ftt / (ft - ftt);
    else z = -1.0f;

    p = {0.0f, 0.0f, 0.0f};
    if (z > 0.0f) p = origin + direction * z;

    int i = u + v * K.width;
    vmap.data[i] = p;
    nmap.data[i] = grad_tsdf(vol, p);
}


__global__
void icp_p2p_se3_kernel(image<JtJse3> JTJ, image<float3> vmap0, image<float3> nmap0, image<float3> vmap1, image<float3> nmap1,
                        intrinsics K, mat4x4 T, float dist_threshold, float angle_threshold)
{
    int u0 = threadIdx.x + blockIdx.x * blockDim.x;
    int v0 = threadIdx.y + blockIdx.y * blockDim.y;
    if (u0 >= K.width || v0 >= K.height) return;

    int i0 = u0 + v0 * K.width;
    JTJ.data[i0].error = 0.0f;
    JTJ.data[i0].weight = 0.0f;

    float3 p0 = T * vmap0.data[i0];
    float3 n0 = rotate(T, nmap0.data[i0]);
    if (p0.z == 0.0f) return;

    int u1 = roundf((p0.x / p0.z) * K.fx + K.cx);
    int v1 = roundf((p0.y / p0.z) * K.fy + K.cy);
    if (u1 < 0 || u1 >= K.width || v1 < 0 || v1 >= K.height) return;

    int i1 = u1 + v1 * K.width;
    float3 p1 = vmap1.data[i1];
    float3 n1 = nmap1.data[i1];
    if (p1.z == 0.0f) return;

    float3 d = p1 - p0;
    if (length(d) >= dist_threshold) return;
    if (dot(n0, n1) < angle_threshold) return;

    float e = dot(d, n1);
    float3 Jt = n1;
    float3 JR = cross(p0, n1);
    JTJ.data[i0].weight  = 1.0f;
    JTJ.data[i0].error   = e * e;
    JTJ.data[i0].Jte[0]  = e * Jt.x;
    JTJ.data[i0].Jte[1]  = e * Jt.y;
    JTJ.data[i0].Jte[2]  = e * Jt.z;
    JTJ.data[i0].Jte[3]  = e * JR.x;
    JTJ.data[i0].Jte[4]  = e * JR.y;
    JTJ.data[i0].Jte[5]  = e * JR.z;
    JTJ.data[i0].JtJ[0]  = Jt.x * Jt.x;
    JTJ.data[i0].JtJ[1]  = Jt.x * Jt.y;
    JTJ.data[i0].JtJ[2]  = Jt.x * Jt.z;
    JTJ.data[i0].JtJ[3]  = Jt.x * JR.x;
    JTJ.data[i0].JtJ[4]  = Jt.x * JR.y;
    JTJ.data[i0].JtJ[5]  = Jt.x * JR.z;
    JTJ.data[i0].JtJ[6]  = Jt.y * Jt.y;
    JTJ.data[i0].JtJ[7]  = Jt.y * Jt.z;
    JTJ.data[i0].JtJ[8]  = Jt.y * JR.x;
    JTJ.data[i0].JtJ[9]  = Jt.y * JR.y;
    JTJ.data[i0].JtJ[10] = Jt.y * JR.z;
    JTJ.data[i0].JtJ[11] = Jt.z * Jt.z;
    JTJ.data[i0].JtJ[12] = Jt.z * JR.x;
    JTJ.data[i0].JtJ[13] = Jt.z * JR.y;
    JTJ.data[i0].JtJ[14] = Jt.z * JR.z;
    JTJ.data[i0].JtJ[15] = JR.x * JR.x;
    JTJ.data[i0].JtJ[16] = JR.x * JR.y;
    JTJ.data[i0].JtJ[17] = JR.x * JR.z;
    JTJ.data[i0].JtJ[18] = JR.y * JR.y;
    JTJ.data[i0].JtJ[19] = JR.y * JR.z;
    JTJ.data[i0].JtJ[20] = JR.z * JR.z;
}


__global__
void se3_reduce_kernel(image<JtJse3> JTJ, image<JtJse3> Axb)
{
    extern __shared__ JtJse3 data[];
    int tid = threadIdx.x;

    JtJse3 r;
    for (int v = blockIdx.x; v < JTJ.height; v += gridDim.x) {
        for (int u = threadIdx.x; u < JTJ.width; u += blockDim.x) {
            int i = u + v * JTJ.width;
            if (JTJ.data[i].weight == 0) continue;
            r += JTJ.data[i];
        }
    }
    data[tid] = r;
    __syncthreads();

    if (tid > 0) return;
    for (int i = 1; i < blockDim.x; ++i)
        data[0] += data[i];
    Axb.data[blockIdx.x] = data[0];
}


static void compute_depth_map(const image<uint16_t>* rmap, image<float>* dmap, intrinsics K, float cutoff)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_depth_kernel<<<grid_size, block_size>>>(rmap->gpu(), dmap->gpu(), K, cutoff);
}


static void compute_vertex_map(const image<float>* dmap, image<float3>* vmap, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_vertex_kernel<<<grid_size, block_size>>>(dmap->gpu(), vmap->gpu(), K);
}


static void compute_normal_map(const image<float3>* vmap, image<float3>* nmap, intrinsics K)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    compute_normal_kernel<<<grid_size, block_size>>>(vmap->gpu(), nmap->gpu(), K);
}


static void integrate_volume(const volume<sdf32f_t>* vol, image<float>* dmap, intrinsics K, mat4x4 P, float mu)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->dimension.x, block_size.x);
    grid_size.y = divup(vol->dimension.y, block_size.y);
    grid_size.z = divup(vol->dimension.z, block_size.z);
    integrate_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), dmap->gpu(), K, P, mu);
}


static void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vmap, image<float3>* nmap, intrinsics K, mat4x4 P, float near, float far)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    raycast_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), vmap->gpu(), nmap->gpu(), K, P, near, far);
}


static mat4x4 icp_p2p_se3(image<float3>* vmap0, image<float3>* nmap0, image<float3>* vmap1, image<float3>* nmap1,
                          intrinsics K, mat4x4 T, float dist_threshold, float angle_threshold)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);

    image<JtJse3> JTJ, Axb;
    unsigned int reduce_size = 8;
    unsigned int reduce_threads = 256;
    JTJ.resize(K.width, K.height, ALLOCATOR_DEVICE);
    Axb.resize(reduce_size, 1, ALLOCATOR_MAPPED);

    icp_p2p_se3_kernel<<<grid_size, block_size>>>(JTJ.gpu(), vmap0->gpu(), nmap0->gpu(), vmap1->gpu(), nmap1->gpu(), K, T, dist_threshold, angle_threshold);
    se3_reduce_kernel<<<reduce_size, reduce_threads, reduce_threads * sizeof(JtJse3)>>>(JTJ.gpu(), Axb.gpu());
    cudaDeviceSynchronize();

    for (int i = 1; i < reduce_size; ++i)
        Axb.data[0] += Axb.data[i];

    JTJ.deallocate();
    Axb.deallocate();
    return T;
}


static void extract_isosurface_cloud(const volume<sdf32f_t>* vol, point_cloud* pc)
{
}


pipeline::pipeline()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}


pipeline::~pipeline()
{
    rmap.deallocate();
    dmap.deallocate();
    cmap.deallocate();
    vmap.deallocate();
    nmap.deallocate();
    rvmap.deallocate();
    rnmap.deallocate();
}


void pipeline::process()
{
    cam->read(&rmap, &cmap);
    preprocess();
    if (frame > 0) track();
    integrate();
    raycast();
    frame++;
}


void pipeline::preprocess()
{
    dmap.resize(rmap.width, rmap.height, ALLOCATOR_DEVICE);
    vmap.resize(rmap.width, rmap.height, ALLOCATOR_DEVICE);
    nmap.resize(rmap.width, rmap.height, ALLOCATOR_DEVICE);

    compute_depth_map(&rmap, &dmap, cam->K, cutoff);
    compute_vertex_map(&dmap, &vmap, cam->K);
    compute_normal_map(&vmap, &nmap, cam->K);
}


void pipeline::integrate()
{
    integrate_volume(vol, &dmap, cam->K, P, mu);
}


void pipeline::raycast()
{
    rvmap.resize(vmap.width, vmap.height, ALLOCATOR_DEVICE);
    rnmap.resize(nmap.width, nmap.height, ALLOCATOR_DEVICE);
    raycast_volume(vol, &rvmap, &rnmap, cam->K, P, near, far);
}


void pipeline::track()
{
    mat4x4 T = P;

    mat4x4 delta = icp_p2p_se3(&vmap, &nmap, &rvmap, &rnmap, cam->K, T, dist_threshold, angle_threshold);
    T = delta * T;

    P = T;
}


void pipeline::extract_point_cloud(point_cloud* pc)
{
    extract_isosurface_cloud(vol, pc);
}
