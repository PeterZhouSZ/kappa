#pragma once
#include "camera.hpp"
#include "common.hpp"
#include "cloud.hpp"
#include "math.hpp"
#include "volume.hpp"


void raw_to_depth(const image<uint16_t>* rm, image<float>* dm, intrinsics K, float cutoff);
void depth_to_vertex(const image<float>* dm, image<float3>* vm, intrinsics K);
void vertex_to_normal(const image<float3>* vm, image<float4>* nm, intrinsics K);
void vertex_to_normal_radius(const image<float3>* vm, image<float4>* nm, intrinsics K);
void depth_bilateral(const image<float>* dm0, image<float>* dm1, intrinsics K, float d_sigma, float r_sigma);

void reset_volume(volume<sdf32f_t>* vol);
void integrate_volume(volume<sdf32f_t>* vol, image<float>* dm, intrinsics K, mat4x4 T, float mu, float maxw);
void raycast_volume(const volume<sdf32f_t>* vol, image<float3>* vm, image<float4>* nm, intrinsics K, mat4x4 T, float mu, float near, float far);

void reset_cloud(cloud<surfel32f_t>* pc);
void integrate_cloud(cloud<surfel32f_t>* pc, image<float3>* vm, image<float4>* nm, image<uint4>* im, intrinsics K, mat4x4 T);
void raycast_cloud(const cloud<surfel32f_t>* pc, image<float3>* vm, image<float4>* nm, image<uint4>* im, intrinsics K, mat4x4 T);

mat4x4 icp_p2p_se3(image<float3>* vm0, image<float4>* nm0, image<float3>* vm1, image<float4>* nm1,
                   intrinsics K, mat4x4 T, int num_iterations, float dist_threshold, float angle_threshold);

void render_phong_light(image<rgb8_t>* im, const image<float3>* vm, const image<float4>* nm, intrinsics K, float3 light, float3 view);
void render_normal(image<rgb8_t>* im, const image<float4>* nm, intrinsics K);

uint32_t sum_scan_cuda(uint32_t* a, uint32_t* sum, int n);
