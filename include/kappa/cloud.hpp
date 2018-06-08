#pragma once
#include <cuda_runtime_api.h>


struct surfel32f_t {
    float3 pos;
    float4 normal;
    float weight;
};


template <typename T>
struct cloud {
    cloud() = default;
    ~cloud() = default;

    void allocate(int size, int device = DEVICE_CUDA);
    void deallocate();

    cloud<T> gpu() const;

    int size = 0;
    int count = 0;
    int device;
    T* data = NULL;
};


template <typename T>
void cloud<T>::allocate(int size, int device)
{
    this->size = size;
    this->count = 0;
    this->device = device;
    switch (this->device) {
        case DEVICE_CUDA:
            cudaMalloc((void**)(&this->data), sizeof(T) * size);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostAlloc((void**)(&this->data), sizeof(T) * size, cudaHostAllocMapped);
            break;
    }
}


template <typename T>
void cloud<T>::deallocate()
{
    switch (this->device) {
        case DEVICE_CUDA:
            cudaFree(this->data);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaFreeHost(this->data);
            break;
    }
    this->size = 0;
    this->count = 0;
    this->data = NULL;
}
