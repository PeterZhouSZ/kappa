#pragma once
#include "common.hpp"


template <typename T>
struct array {
    array() = default;
    ~array() = default;

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

    void alloc(int size, int device);
    void free();
    void resize(int size);
    void clear(uint8_t c = 0x00);

    array<T> cpu() const;
    array<T> cuda() const;

    int size = 0;
    int device;
    T* data = nullptr;
};


template <typename T>
void array<T>::alloc(int size, int device)
{
    this->size = size;
    this->device = device;
    switch (device) {
        case DEVICE_CPU:
            data = (T*)malloc(sizeof(T) * size);
            break;
        case DEVICE_CUDA:
            cudaMalloc((void**)&data, sizeof(T) * size);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostAlloc((void**)&data, sizeof(T) * size, cudaHostAllocMapped);
            break;
    }
}


template <typename T>
void array<T>::free()
{
    switch (device) {
        case DEVICE_CUDA:
            cudaFree(data);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaFreeHost(data);
            break;
    }
    size = 0;
    data = nullptr;
}


template <typename T>
array<T> array<T>::cuda() const
{
    array<T> arr;
    arr.size = size;
    arr.device = device;
    switch (device) {
        case DEVICE_CPU:
            arr.data = nullptr;
            break;
        case DEVICE_CUDA:
            arr.data = data;
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostGetDevicePointer(&arr.data, data, 0);
            break;
    }
    return arr;
}
