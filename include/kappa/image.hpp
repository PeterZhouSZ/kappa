#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "common.hpp"


template <typename T>
struct image {
    image() = default;
    ~image() = default;

    void resize(int width, int height, int device = DEVICE_CPU);
    void allocate(int width, int height, int device = DEVICE_CPU);
    void deallocate();
    void clear();

    image<T> gpu() const;

    int width = 0;
    int height = 0;
    int device;
    T* data = NULL;
};


template <typename T>
void image<T>::resize(int width, int height, int device)
{
    if (this->width == width && this->height == height) return;
    deallocate();
    allocate(width, height, device);
}


template <typename T>
void image<T>::allocate(int width, int height, int device)
{
    this->width = width;
    this->height = height;
    this->device = device;
    size_t size = width * height * sizeof(T);
    switch (device) {
        case DEVICE_CPU:
            this->data = (T*)malloc(size);
            break;
        case DEVICE_CUDA:
            cudaMalloc((void**)(&data), size);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostAlloc((void**)(&data), size, cudaHostAllocMapped);
            break;
    }
}


template <typename T>
void image<T>::deallocate()
{
    switch (this->device) {
        case DEVICE_CPU:
            free(this->data);
            break;
        case DEVICE_CUDA:
            cudaFree(this->data);
            break;
        case DEVICE_CUDA_MAPPED:
            cudaFreeHost(this->data);
            break;
    }
    this->width = 0;
    this->height = 0;
    this->data = NULL;
}


template <typename T>
void image<T>::clear()
{
    size_t size = width * height * sizeof(T);
    switch (this->device) {
        case DEVICE_CPU:
            memset(this->data, 0, size);
            break;
        case DEVICE_CUDA:
        case DEVICE_CUDA_MAPPED:
            cudaMemset(this->data, 0, size);
            break;
    }
}


template <typename T>
image<T> image<T>::gpu() const
{
    image<T> im;
    im.width = this->width;
    im.height = this->height;
    im.device = DEVICE_CUDA;
    switch (this->device) {
        case DEVICE_CPU:
            im.data = NULL;
            break;
        case DEVICE_CUDA:
            im.data = this->data;
            break;
        case DEVICE_CUDA_MAPPED:
            cudaHostGetDevicePointer(&im.data, this->data, 0);
            break;
    }
    return im;
}
