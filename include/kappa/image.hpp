#pragma once
#include "common.hpp"


template <typename T>
struct image {
    image() = default;
    ~image() = default;

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

    void alloc(int width, int height, int device);
    void free();
    void resize(int width, int height, int device);
    void clear(uint8_t c = 0x00);

    image<T> cpu() const;
    image<T> cuda() const;

    int width = 0;
    int height = 0;
    int device;
    T* data = nullptr;
};


template <typename T>
void image<T>::resize(int width, int height, int device)
{
    int size = this->width * this->height;
    if (size > width * height) {
        this->width = width;
        this->height = height;
        return;
    }
    free();
    alloc(width, height, device);
}


template <typename T>
void image<T>::alloc(int width, int height, int device)
{
    this->width = width;
    this->height = height;
    this->device = device;
    size_t size = width * height * sizeof(T);
    switch (device) {
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
void image<T>::free()
{
    switch (device) {
        case DEVICE_CPU:
            ::free(data);
            break;
        case DEVICE_CUDA:
            cudaFree(data);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaFreeHost(data);
            break;
    }
    width = 0;
    height = 0;
    data = nullptr;
}


template <typename T>
void image<T>::clear(uint8_t c)
{
    size_t size = width * height * sizeof(T);
    switch (this->device) {
        case DEVICE_CPU:
            memset(data, c, size);
            break;
        case DEVICE_CUDA:
        case DEVICE_CUDA_MAPPED:
            cudaMemset(data, c, size);
            break;
    }
}


template <typename T>
image<T> image<T>::cuda() const
{
    image<T> im;
    im.width = width;
    im.height = height;
    im.device = DEVICE_CUDA;
    switch (device) {
        case DEVICE_CPU:
            im.data = nullptr;
            break;
        case DEVICE_CUDA:
            im.data = data;
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostGetDevicePointer(&im.data, data, 0);
            break;
    }
    return im;
}
