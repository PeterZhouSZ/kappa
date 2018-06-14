#pragma once
#include <cuda_runtime_api.h>


struct surfel32f_t {
    float3 pos;
    float3 normal;
    float radius;
    float weight;
};


template <typename T>
struct cloud {
    cloud() = default;
    ~cloud() = default;

    void allocate(int capacity, int device = DEVICE_CUDA);
    void deallocate();

    cloud<T> gpu() const;

    int capacity = 0;
    int size = 0;
    int device;
    T* data = NULL;
};


template <typename T>
void cloud<T>::allocate(int capacity, int device)
{
    this->capacity = capacity;
    this->size = 0;
    this->device = device;
    switch (this->device) {
        case DEVICE_CUDA:
            cudaMalloc((void**)(&this->data), sizeof(T) * capacity);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostAlloc((void**)(&this->data), sizeof(T) * capacity, cudaHostAllocMapped);
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
    this->capacity = 0;
    this->size = 0;
    this->data = NULL;
}


template <typename T>
cloud<T> cloud<T>::gpu() const
{
    cloud<T> pc;
    pc.capacity = this->capacity;
    pc.size = this->size;
    pc.device = DEVICE_CUDA;
    switch (this->device) {
        case DEVICE_CPU:
            pc.data = NULL;
            break;
        case DEVICE_CUDA:
            pc.data = this->data;
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostGetDevicePointer(&pc.data, this->data, 0);
            break;
    }
    return pc;
}
