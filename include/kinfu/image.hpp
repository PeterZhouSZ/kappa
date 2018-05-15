#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "common.hpp"


struct rgb8_t { uint8_t r, g, b; };

template <typename T>
struct image {
    image() = default;
    ~image() = default;

    void resize(int width, int height, allocator alloc = ALLOCATOR_MAPPED);
    void allocate(int width, int height, allocator alloc = ALLOCATOR_MAPPED);
    void deallocate();

    image<T> gpu() const;
    image<T> cpu() const;

    int width = 0;
    int height = 0;
    T* data = NULL;
    allocator alloc;
};


template <typename T>
void image<T>::resize(int width, int height, allocator alloc)
{
    if (this->width == width && this->height == height) return;
    deallocate();
    allocate(width, height, alloc);
}


template <typename T>
void image<T>::allocate(int width, int height, allocator alloc)
{
    this->width = width;
    this->height = height;
    this->alloc = alloc;
    size_t size = width * height * sizeof(T);
    switch (alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            cudaMalloc((void**)(&data), size);
            break;
        case ALLOCATOR_MAPPED:
            cudaHostAlloc((void**)(&data), size, cudaHostAllocMapped);
            break;
    }
}


template <typename T>
void image<T>::deallocate()
{
    switch (this->alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            cudaFree(this->data);
            break;
        case ALLOCATOR_MAPPED:
            cudaFreeHost(this->data);
            break;
    };
    this->width = 0;
    this->height = 0;
    this->data = NULL;
}


template <typename T>
image<T> image<T>::gpu() const
{
    image<T> im;
    im.width = this->width;
    im.height = this->height;
    im.alloc = ALLOCATOR_DEVICE;
    switch (this->alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            im.data = this->data;
            break;
        case ALLOCATOR_MAPPED:
            cudaHostGetDevicePointer(&im.data, this->data, 0);
            break;
    }
    return im;
}
