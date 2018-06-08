#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "common.hpp"


struct sdf32f_t {
    float tsdf;
    float weight;
    rgb8_t color;
};


template <typename T>
struct volume {
    volume() = default;
    ~volume() = default;

    void resize(int3 dimension, int device = DEVICE_CUDA);
    void allocate(int3 dimension, int device = DEVICE_CUDA);
    void deallocate();

    volume<T> gpu() const;

    T* data = NULL;
    int3 dimension = {0, 0, 0};
    float3 offset = {0.0f, 0.0f, 0.0f};
    float voxel_size = 0.0f;
    int device;
};

__device__ float tsdf_at(volume<sdf32f_t> vol, int x, int y, int z);
__device__ float nearest_tsdf(volume<sdf32f_t> vol, float3 p);
__device__ float interp_tsdf(volume<sdf32f_t> vol, float3 p);
__device__ float3 grad_tsdf(volume<sdf32f_t> vol, float3 p);


template <typename T>
void volume<T>::allocate(int3 dimension, int device)
{
    this->dimension = dimension;
    this->device = device;
    size_t size = dimension.x * dimension.y * dimension.z * sizeof(T);
    switch (device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            cudaMalloc((void**)&data, size);
            cudaMemset(data, 0, size);
        case DEVICE_CUDA_MAPPED:
            break;
    }
}


template <typename T>
void volume<T>::deallocate()
{
    switch (this->device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            cudaFree(this->data);
            break;
        case DEVICE_CUDA_MAPPED:
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
    vol.device = DEVICE_CUDA;
    switch (this->device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            vol.data = this->data;
            break;
        case DEVICE_CUDA_MAPPED:
            break;
    }
    return vol;
}
