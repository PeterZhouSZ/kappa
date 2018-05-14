#pragma once
#include <cuda_runtime_api.h>


namespace kinfu {

struct point_cloud {
    int size;
    float3* vertices = NULL;
    float3* normals = NULL;
    float3* colors = NULL;
};

}
