#pragma once
#include "camera.hpp"
#include "common.hpp"
#include "image.hpp"



struct renderer {
    renderer() = default;
    ~renderer() = default;

    void render_phong(image<rgb8_t>* im, const image<float3>* vmap, const image<float3>* nmap);

    mat4x4 T;
    intrinsics K;
};
