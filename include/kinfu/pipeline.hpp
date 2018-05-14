#pragma once
#include <vector>
#include "camera.hpp"
#include "common.hpp"
#include "point_cloud.hpp"
#include "volume.hpp"


namespace kinfu {

struct pipeline {
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
    float cutoff = 4.0f;
    float mu = 0.1;
    mat4x4 P;
    std::vector<mat4x4> poses;

    image<uint16_t> dmap;
    image<rgb8_t>   cmap;
    image<float3>   vmap;
    image<float3>   nmap;
    image<float3>   rvmap;
    image<float3>   rnmap;
};

}
