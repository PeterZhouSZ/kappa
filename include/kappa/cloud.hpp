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
    switch (this->device) {
        case DEVICE_CUDA:
            CUDA_MALLOC_T(data, T, capacity);
            break;
        case DEVICE_CUDA_MAPPED:
            CUDA_MALLOC_MAPPED_T(data, T, capacity);
            break;
    }
}


template <typename T>
void cloud<T>::free()
{
    switch (device) {
        case DEVICE_CUDA:
            CUDA_FREE(data);
            break;
        case DEVICE_CUDA_MAPPED:
            CUDA_FREE_MAPPED(data);
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
            CUDA_MAP_PTR(pcd.data, data);
            break;
    }
    return pcd;
}
