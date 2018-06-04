#pragma once
#include <vector>
#include "camera.hpp"
#include "common.hpp"
#include "math.hpp"
#include "volume.hpp"


void compute_depth_map(const image<uint16_t>* rm, image<float>* dm, intrinsics K, float cutoff);
void compute_vertex_map(const image<float>* dm, image<float3>* vm, intrinsics K);
void compute_normal_map(const image<float3>* vm, image<float3>* nm, intrinsics K);
void depth_bilateral(const image<float>* dm0, image<float>* dm1, intrinsics K, float d_sigma, float r_sigma);

void reset_volume(volume<sdf32f_t>* vol);
void integrate_volume(const volume<sdf32f_t>* vol, image<float>* dm, intrinsics K, mat4x4 T, float mu, float max_weight);
void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vm, image<float3>* nm, intrinsics K, mat4x4 T, float near, float far);

mat4x4 icp_p2p_se3(image<float3>* vm0, image<float3>* nm0, image<float3>* vm1, image<float3>* nm1,
                   intrinsics K, mat4x4 T, int num_iterations, float dist_threshold, float angle_threshold);

void render_phong_light(image<rgb8_t>* im, const image<float3>* vm, const image<float3>* nm, intrinsics K);
void render_normal(image<rgb8_t>* im, const image<float3>* nm, intrinsics K);


__device__
inline float tsdf_at(volume<sdf32f_t> vol, int x, int y, int z)
{
    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    if (x < 0 || x >= vol.dimension.x ||
        y < 0 || y >= vol.dimension.y ||
        z < 0 || z >= vol.dimension.z)
        return 1.0f; // cannot interpolate
    return vol.data[i].tsdf;
}


__device__
inline float nearest_tsdf(volume<sdf32f_t> vol, float3 p)
{
    int x = roundf((p.x - vol.offset.x) / vol.voxel_size);
    int y = roundf((p.y - vol.offset.y) / vol.voxel_size);
    int z = roundf((p.z - vol.offset.z) / vol.voxel_size);
    return tsdf_at(vol, x, y, z);
}


__device__
inline float interp_tsdf(volume<sdf32f_t> vol, float3 p)
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
inline float3 grad_tsdf(volume<sdf32f_t> vol, float3 p)
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
