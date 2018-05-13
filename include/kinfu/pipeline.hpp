#pragma once
#include <vector>
#include "camera.hpp"
#include "common.hpp"


namespace kinfu {

struct pipeline {
    static constexpr int MAX_PYRAMID_LEVEL = 3;

    pipeline();
    ~pipeline();

    void register_camera(camera* cam);
    void process();

    void preprocess();
    void integrate();
    void raycast();
    void track();

    camera* cam = nullptr;
    int num_levels = 1;
    float cutoff = 4.0f;
    image<uint16_t> dm[MAX_PYRAMID_LEVEL];
    image<rgb8_t>   cm[MAX_PYRAMID_LEVEL];
    image<float3>   vm[MAX_PYRAMID_LEVEL];
    image<float3>   nm[MAX_PYRAMID_LEVEL];
};

}
