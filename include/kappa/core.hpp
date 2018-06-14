#pragma once
#include "camera.hpp"
#include "common.hpp"
#include "cloud.hpp"
#include "math.hpp"
#include "volume.hpp"


void compute_depth_map(const image<uint16_t>* rm, image<float>* dm, intrinsics K, float cutoff);
void compute_vertex_map(const image<float>* dm, image<float3>* vm, intrinsics K);
void compute_normal_map(const image<float3>* vm, image<float3>* nm, intrinsics K);
void compute_normal_radius_map(const image<float3>* vm, image<float4>* nm, intrinsics K);
void depth_bilateral(const image<float>* dm0, image<float>* dm1, intrinsics K, float d_sigma, float r_sigma);

void reset_volume(volume<sdf32f_t>* vol);
void integrate_volume(volume<sdf32f_t>* vol, image<float>* dm, intrinsics K, mat4x4 T, float mu, float maxw);
void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vm, image<float3>* nm, intrinsics K, mat4x4 T, float mu, float near, float far);

void reset_cloud(cloud<surfel32f_t>* pc);
void integrate_cloud(cloud<surfel32f_t>* pc, image<float3>* vm, image<float4>* nm, image<uint4>* im, intrinsics K, mat4x4 T);

mat4x4 icp_p2p_se3(image<float3>* vm0, image<float3>* nm0, image<float3>* vm1, image<float3>* nm1,
                   intrinsics K, mat4x4 T, int num_iterations, float dist_threshold, float angle_threshold);

void render_phong_light(image<rgb8_t>* im, const image<float3>* vm, const image<float3>* nm, intrinsics K);
void render_normal(image<rgb8_t>* im, const image<float3>* nm, intrinsics K);
