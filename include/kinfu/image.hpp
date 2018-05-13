#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>


namespace kinfu {

enum allocator {
    ALLOCATOR_HOST,
    ALLOCATOR_DEVICE,
    ALLOCATOR_MAPPED
};

struct rgb8_t { uint8_t r, g, b; };

template <typename T>
struct image {
    image() = default;
    ~image() = default;

    void resize(int width, int height, allocator alloc = ALLOCATOR_MAPPED);
    void allocate(int width, int height, allocator alloc = ALLOCATOR_MAPPED);
    void deallocate();

    T* gpu() const;

    int width = 0;
    int height = 0;
    T* data = nullptr;
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
    switch (alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            cudaMalloc((void**)(&data), width * height * sizeof(T));
            break;
        case ALLOCATOR_MAPPED:
            cudaHostAlloc((void**)(&data), width * height * sizeof(T), cudaHostAllocMapped);
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
    this->data = nullptr;
}


template <typename T>
T* image<T>::gpu() const
{
    T* ptr = nullptr;
    switch (this->alloc) {
        case ALLOCATOR_HOST:
            break;
        case ALLOCATOR_DEVICE:
            ptr = this->data;
            break;
        case ALLOCATOR_MAPPED:
            cudaHostGetDevicePointer(&ptr, this->data, 0);
            break;
    }
    return ptr;
}

}
