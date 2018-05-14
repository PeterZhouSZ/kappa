#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "common.hpp"


namespace kinfu {

struct sdf32f_t {
    float tsdf;
    uint8_t weight;
};

template <typename T>
struct volume {
    volume() = default;
    ~volume() = default;

    void resize(int3 dimension, allocator alloc = ALLOCATOR_DEVICE);
    void allocate(int3 dimension, allocator alloc = ALLOCATOR_DEVICE);
    void deallocate();

    volume<T> gpu() const;
    volume<T> cpu() const;

    T* data = NULL;
    int3 dimension = {0, 0, 0};
    float3 offset = {0.0f, 0.0f, 0.0f};
    float voxel_size = 0.0f;
    allocator alloc;
};


template <typename T>
void volume<T>::allocate(int3 dimension, allocator alloc)
{
    this->dimension = dimension;
    this->alloc = alloc;
    size_t size = dimension.x * dimension.y * dimension.z * sizeof(T);
    switch (alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            cudaMalloc((void**)&data, size);
            cudaMemset(data, 0, size);
        case ALLOCATOR_MAPPED:
            break;
    }
}


template <typename T>
void volume<T>::deallocate()
{
    switch (this->alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            cudaFree(this->data);
            break;
        case ALLOCATOR_MAPPED:
            break;
    };
    data = NULL;
    dimension = {0, 0, 0};
}


template <typename T>
volume<T> volume<T>::gpu() const
{
    volume<T> vol;
    vol.dimension = this->dimension;
    vol.offset = this->offset;
    vol.voxel_size = this->voxel_size;
    vol.alloc = ALLOCATOR_DEVICE;
    switch (this->alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            vol.data = this->data;
            break;
        case ALLOCATOR_MAPPED:
            break;
    }
    return vol;
}

}
