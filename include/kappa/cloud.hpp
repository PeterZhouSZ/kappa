#pragma once
#include "common.hpp"


template <typename T>
struct cloud {
    cloud() = default;
    ~cloud() = default;

    CPU_GPU_CODE
    T& operator[](int i)
    {
        return data[i];
    }

    CPU_GPU_CODE
    const T& operator[](int i) const
    {
        return data[i];
    }

    void alloc(int capacity, int device);
    void free();

    cloud<T> cpu() const;
    cloud<T> cuda() const;

    int capacity = 0;
    int size = 0;
    int device;
    T* data = nullptr;
};


template <typename T>
void cloud<T>::alloc(int capacity, int device)
{
    this->capacity = capacity;
    this->size = 0;
    this->device = device;
    size_t size = sizeof(T) * capacity;
    switch (this->device) {
        case DEVICE_CPU:
            data = (T*)malloc(size);
            break;
        case DEVICE_CUDA:
            cudaMalloc((void**)&data, size);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostAlloc((void**)&data, size, cudaHostAllocMapped);
            break;
    }
}


template <typename T>
void cloud<T>::free()
{
    switch (device) {
        case DEVICE_CUDA:
            cudaFree(data);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaFreeHost(data);
            break;
    }
    capacity = 0;
    size = 0;
    data = nullptr;
}


template <typename T>
cloud<T> cloud<T>::cuda() const
{
    cloud<T> pcd;
    pcd.capacity = capacity;
    pcd.size = size;
    pcd.device = DEVICE_CUDA;
    switch (this->device) {
        case DEVICE_CPU:
            pcd.data = nullptr;
            break;
        case DEVICE_CUDA:
            pcd.data = data;
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostGetDevicePointer(&pcd.data, data, 0);
            break;
    }
    return pcd;
}
