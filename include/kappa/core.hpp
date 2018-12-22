#pragma once
#include "camera.hpp"
#include "common.hpp"
#include "cloud.hpp"
#include "math.hpp"
#include "volume.hpp"


void raw_to_depth(
    const image<uint16_t> rm,
    image<float>* dm,
    intrinsics K,
    float cutoff);

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

void reset(volume<voxel>* vol);

void integrate(volume<voxel>* vol,
               const image<float> dm,
               intrinsics K,
               mat4x4 T,
               float mu,
               float maxw);

void raycast(const volume<voxel> vol,
             image<float3>* vm,
             image<float4>* nm,
             intrinsics K,
             mat4x4 T,
             float mu,
             float near,
             float far);

void reset(cloud<surfel>* pcd);

void integrate(cloud<surfel>* pcd,
               const image<float3> vm,
               const image<float4> nm,
               const image<uint32_t> im,
               intrinsics K,
               mat4x4 T);

void raycast(const cloud<surfel> pcd,
             image<float3>* vm,
             image<float4>* nm,
             image<uint32_t>* im,
             intrinsics K,
             mat4x4 T);

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

void render_normal(
    const image<float3> vm,
    const image<float4> nm,
    image<rgb8>* im,
    intrinsics K);

uint32_t prescan(uint32_t* a, uint32_t* sum, int n);
