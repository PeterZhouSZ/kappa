#include <kinfu/pipeline.hpp>
#include <float.h>
#include <Eigen/SVD>
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


__global__
void integrate_volume_kernel(volume<sdf32f_t> vol, image<float> dmap, intrinsics K, mat4x4 T, float mu)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= vol.dimension.x || y >= vol.dimension.y || z >= vol.dimension.z) return;

    float3 p ={(float)x, (float)y, (float)z};
    p = vol.offset + p * vol.voxel_size;
    float3 q = T * p;
    if (q.z <= 0.001f) return;

    int u = roundf((q.x / q.z) * K.fx + K.cx);
    int v = roundf((q.y / q.z) * K.fy + K.cy);
    if (u < 0 || u >= K.width || v < 0 || v >= K.height) return;

    float d = dmap.data[u + v * K.width];
    if (d == 0.0f) return;

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
void raycast_volume_kernel(volume<sdf32f_t> vol, image<float3> vmap, image<float3> nmap, intrinsics K, mat4x4 T, float near, float far)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    float3 q;
    q.x = (u - K.cx) / K.fx;
    q.y = (v - K.cy) / K.fy;
    q.z = 1.0f;

    float3 origin = {T.m03, T.m13, T.m23};
    float3 direction = rotate(T, q);

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

    int i = u + v * K.width;
    vmap.data[i] = {0.0f, 0.0f, 0.0f};
    nmap.data[i] = {0.0f, 0.0f, 0.0f};
    if (z >= 0.0f) {
        p = origin + direction * z;
        vmap.data[i] = p;
        nmap.data[i] = grad_tsdf(vol, p);
    }
}


__global__
void icp_p2p_se3_kernel(image<JtJse3> JTJ, image<float3> vmap0, image<float3> nmap0, image<float3> vmap1, image<float3> nmap1,
                        intrinsics K, mat4x4 T, float dist_threshold, float angle_threshold)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    JTJ.data[i].error = 0.0f;
    JTJ.data[i].weight = 0.0f;

    float3 p0 = T * vmap0.data[i];
    float3 n0 = rotate(T, nmap0.data[i]);
    if (p0.z == 0.0f) return;

    float3 p1 = vmap1.data[i];
    float3 n1 = nmap1.data[i];
    if (p1.z == 0.0f) return;

    float3 d = p1 - p0;
    if (length(d) >= dist_threshold) return;
    if (fabs(dot(n0, n1)) < angle_threshold) return;

    float e = dot(d, n1);
    float3 Jt = n1;
    float3 JR = cross(p0, n1);
    JTJ.data[i].weight  = 1.0f;
    JTJ.data[i].error   = e * e;
    JTJ.data[i].Jte[0]  = e * Jt.x;
    JTJ.data[i].Jte[1]  = e * Jt.y;
    JTJ.data[i].Jte[2]  = e * Jt.z;
    JTJ.data[i].Jte[3]  = e * JR.x;
    JTJ.data[i].Jte[4]  = e * JR.y;
    JTJ.data[i].Jte[5]  = e * JR.z;
    JTJ.data[i].JtJ[0]  = Jt.x * Jt.x;
    JTJ.data[i].JtJ[1]  = Jt.x * Jt.y;
    JTJ.data[i].JtJ[2]  = Jt.x * Jt.z;
    JTJ.data[i].JtJ[3]  = Jt.x * JR.x;
    JTJ.data[i].JtJ[4]  = Jt.x * JR.y;
    JTJ.data[i].JtJ[5]  = Jt.x * JR.z;
    JTJ.data[i].JtJ[6]  = Jt.y * Jt.y;
    JTJ.data[i].JtJ[7]  = Jt.y * Jt.z;
    JTJ.data[i].JtJ[8]  = Jt.y * JR.x;
    JTJ.data[i].JtJ[9]  = Jt.y * JR.y;
    JTJ.data[i].JtJ[10] = Jt.y * JR.z;
    JTJ.data[i].JtJ[11] = Jt.z * Jt.z;
    JTJ.data[i].JtJ[12] = Jt.z * JR.x;
    JTJ.data[i].JtJ[13] = Jt.z * JR.y;
    JTJ.data[i].JtJ[14] = Jt.z * JR.z;
    JTJ.data[i].JtJ[15] = JR.x * JR.x;
    JTJ.data[i].JtJ[16] = JR.x * JR.y;
    JTJ.data[i].JtJ[17] = JR.x * JR.z;
    JTJ.data[i].JtJ[18] = JR.y * JR.y;
    JTJ.data[i].JtJ[19] = JR.y * JR.z;
    JTJ.data[i].JtJ[20] = JR.z * JR.z;
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


static void bilateral_filter(const image<float>* dmap0, image<float>* dmap1, intrinsics K, float d_sigma, float r_sigma)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    bilateral_filter_kernel<<<grid_size, block_size>>>(dmap0->gpu(), dmap1->gpu(), K, d_sigma, r_sigma);
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


static void integrate_volume(const volume<sdf32f_t>* vol, image<float>* dmap, intrinsics K, mat4x4 T, float mu)
{
    dim3 block_size(8, 8, 8);
    dim3 grid_size;
    grid_size.x = divup(vol->dimension.x, block_size.x);
    grid_size.y = divup(vol->dimension.y, block_size.y);
    grid_size.z = divup(vol->dimension.z, block_size.z);
    integrate_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), dmap->gpu(), K, T, mu);
}


static void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vmap, image<float3>* nmap, intrinsics K, mat4x4 T, float near, float far)
{
    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);
    raycast_volume_kernel<<<grid_size, block_size>>>(vol->gpu(), vmap->gpu(), nmap->gpu(), K, T, near, far);
}


static Eigen::Matrix3f rodrigues(float3 w)
{
    float theta = length(w);
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f R = I;

    if (theta >= FLT_EPSILON) {
        float c = cos(theta);
        float s = sin(theta);
        float inv_theta = theta != 0 ? 1.0f / theta : 0.0f;

        Eigen::Matrix3f G;
        G << 0.0f, -w.z,  w.y,
              w.z, 0.0f, -w.x,
             -w.y,  w.x, 0.0f;
        R = I + inv_theta * s * G
            + (1 - c) * inv_theta * inv_theta * G * G;
    }

    return R;
}


static mat4x4 solve_icp_p2p(JtJse3 J)
{
    Eigen::Matrix<float, 6, 6> A;
    Eigen::Matrix<float, 6, 1> b;
    Eigen::Matrix<float, 6, 1> x;

    b(0) = J.Jte[0];
    b(1) = J.Jte[1];
    b(2) = J.Jte[2];
    b(3) = J.Jte[3];
    b(4) = J.Jte[4];
    b(5) = J.Jte[5];

    A(0, 0) = J.JtJ[0];
    A(0, 1) = A(1, 0) = J.JtJ[1];
    A(0, 2) = A(2, 0) = J.JtJ[2];
    A(0, 3) = A(3, 0) = J.JtJ[3];
    A(0, 4) = A(4, 0) = J.JtJ[4];
    A(0, 5) = A(5, 0) = J.JtJ[5];
    A(1, 1) = J.JtJ[6];
    A(1, 2) = A(2, 1) = J.JtJ[7];
    A(1, 3) = A(3, 1) = J.JtJ[8];
    A(1, 4) = A(4, 1) = J.JtJ[9];
    A(1, 5) = A(5, 1) = J.JtJ[10];
    A(2, 2) = J.JtJ[11];
    A(2, 3) = A(3, 2) = J.JtJ[12];
    A(2, 4) = A(4, 2) = J.JtJ[13];
    A(2, 5) = A(5, 2) = J.JtJ[14];
    A(3, 3) = J.JtJ[15];
    A(3, 4) = A(4, 3) = J.JtJ[16];
    A(3, 5) = A(5, 3) = J.JtJ[17];
    A(4, 4) = J.JtJ[18];
    A(4, 5) = A(5, 4) = J.JtJ[19];
    A(5, 5) = J.JtJ[20];

    Eigen::JacobiSVD<Eigen::Matrix<float, 6, 6>>
        svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.solve(b);

    Eigen::Matrix3f R = rodrigues({x(3), x(4), x(5)});
    Eigen::Vector3f t = {x(0), x(1), x(2)};

    return mat4x4{R, t};
}


static mat4x4 icp_p2p_se3(image<float3>* vmap0, image<float3>* nmap0, image<float3>* vmap1, image<float3>* nmap1,
                          intrinsics K, mat4x4 T, float dist_threshold, float angle_threshold, float& error)
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

    mat4x4 delta;
    error = Axb.data[0].error / Axb.data[0].weight;
    if (!isnan(error)) delta = solve_icp_p2p(Axb.data[0]);

    JTJ.deallocate();
    Axb.deallocate();
    return delta;
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
    for (int level = 0; level < num_levels; ++level) {
        dmaps[level].deallocate();
        vmaps[level].deallocate();
        nmaps[level].deallocate();
        rvmaps[level].deallocate();
        rnmaps[level].deallocate();
    }
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
    dmap.resize(cam->K.width, cam->K.height, ALLOCATOR_DEVICE);
    for (int level = 0; level < num_levels; ++level) {
        int width = cam->K.width >> level;
        int height = cam->K.height >> level;
        dmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        vmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        nmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        rvmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        rnmaps[level].resize(width, height, ALLOCATOR_DEVICE);
    }

    compute_depth_map(&rmap, &dmap, cam->K, cutoff);
    bilateral_filter(&dmap, &dmaps[0], cam->K, bilateral_d_sigma, bilateral_r_sigma);
    compute_vertex_map(&dmaps[0], &vmaps[0], cam->K);
    compute_normal_map(&vmaps[0], &nmaps[0], cam->K);
}


void pipeline::integrate()
{
    integrate_volume(vol, &dmap, cam->K, P.inverse(), mu);
}


void pipeline::raycast()
{
    raycast_volume(vol, &rvmaps[0], &rnmaps[0], cam->K, P, near, far);
}


void pipeline::track()
{
    mat4x4 T = P;
    float last_error = FLT_MAX;
    for (int i = 0; i < icp_num_iterations; ++i) {
        float error;
        mat4x4 delta = icp_p2p_se3(&vmaps[0], &nmaps[0], &rvmaps[0], &rnmaps[0], cam->K, T, dist_threshold, angle_threshold, error);
        printf("%f\n", error);
        if (isnan(error) || error > last_error) break;
        T = delta * T;
        last_error = error;
    }
    P = T;
    print_mat4x4(stdout, P);
    printf("%d\n", frame);
    printf("\n");
}


void pipeline::extract_point_cloud(point_cloud* pc)
{
    extract_isosurface_cloud(vol, pc);
}
