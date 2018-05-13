#pragma once
#include <vector>
#include "camera.hpp"
#include "common.hpp"


namespace kinfu {

struct pipeline {
    static constexpr int max_pyramid_level = 3;

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
    image<uint16_t> dm[max_pyramid_level];
    image<rgb8_t>   cm[max_pyramid_level];
    image<float3>   vm[max_pyramid_level];
};

}
