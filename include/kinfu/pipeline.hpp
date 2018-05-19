#pragma once
#include <vector>
#include "camera.hpp"
#include "common.hpp"
#include "point_cloud.hpp"
#include "volume.hpp"


struct pipeline {
    static constexpr int num_levels = 3;

    pipeline();
    ~pipeline();

    void process();

    void preprocess();
    void integrate();
    void raycast();
    void track();
    void extract_point_cloud(point_cloud* pc);

    volume<sdf32f_t>* vol = NULL;

    camera* cam = NULL;
    int frame = 0;
    int icp_num_iterations = 1;
    float ratio_threshold = 0.5f;
    float dist_threshold = 0.05f;
    float angle_threshold = 0.8f;
    float cutoff = 4.0f;
    float near = 0.001f;
    float far = 4.0f;
    float mu = 0.1f;
    mat4x4 P;
    std::vector<mat4x4> poses;

    image<uint16_t> rmap;
    image<float>    dmap;
    image<float>    dmaps[num_levels];
    image<rgb8_t>   cmap;
    image<float3>   vmap;
    image<float3>   nmap;
    image<uint8_t>  tmap;
    image<float3>   rvmap;
    image<float3>   rnmap;
};
