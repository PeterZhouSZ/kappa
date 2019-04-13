#pragma once
#include "array.hpp"
#include "camera.hpp"
#include "cloud.hpp"
#include "io.hpp"
#include "math.hpp"
#include "volume.hpp"
#include "sequence.hpp"


void raw_to_depth(
    const image<uint16_t> rdm,
    image<float>* dm,
    intrinsics K,
    float factor,
    float cutoff);

void raw_to_color(
    const image<rgb8> rcm,
    image<float3>* cm,
    intrinsics K);

void depth_to_vertex(
    const image<float> dm,
    image<float3>* vm,
    intrinsics K);

void vertex_to_normal(
    const image<float3> vm,
    image<float4>* nm,
    intrinsics K);

void vertex_to_normal_radius(
    const image<float3> vm,
    image<float4>* nm,
    intrinsics K);

void depth_bilateral(
    const image<float> dm0,
    image<float>* dm1,
    intrinsics K,
    float d_sigma,
    float r_sigma);

void reset_volume(volume<voxel>* vol);

void integrate_volume(
    volume<voxel>* vol,
    const image<float> dm,
    const image<float3> cm,
    intrinsics K,
    mat4x4 T,
    float mu,
    float maxw);

void raycast_volume(
    const volume<voxel> vol,
    image<float3>* vm,
    image<float4>* nm,
    image<float3>* cm,
    intrinsics K,
    mat4x4 T,
    float mu,
    float near,
    float far);

void reset_cloud(cloud<surfel>* pcd);

void integrate_cloud(
    cloud<surfel>* pcd,
    const image<float3> vm,
    const image<float4> nm,
    const image<float3> cm,
    const image<uint32_t> idm,
    intrinsics K,
    mat4x4 T,
    int timestamp,
    float delta_r);

void raycast_cloud(
    const cloud<surfel> pcd,
    image<float3>* vm,
    image<float4>* nm,
    image<float3>* cm,
    image<uint32_t>* idm,
    intrinsics K,
    mat4x4 T,
    int timestamp,
    float maxw,
    float cutoff);

void cleanup_cloud(
    cloud<surfel>* pcd,
    float maxw,
    int timestamp,
    int delta_t);

mat4x4 icp_p2p_se3(
    const image<float3> vm0,
    const image<float4> nm0,
    const image<float3> vm1,
    const image<float4> nm1,
    intrinsics K,
    mat4x4 T,
    int num_iterations,
    float dist_threshold,
    float angle_threshold);

void render_phong_light(
    const image<float3> vm,
    const image<float4> nm,
    image<rgb8>* im,
    intrinsics K,
    float3 light,
    float3 view);

void render_phong_light(
    const image<float3> vm,
    const image<float4> nm,
    const image<float3> cm,
    image<rgb8>* im,
    intrinsics K,
    float3 light,
    float3 view);

uint32_t prescan(uint32_t* a, uint32_t* sum, int n);
