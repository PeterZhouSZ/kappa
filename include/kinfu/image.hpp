#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>


namespace kinfu {

struct rgb8_t { uint8_t r, g, b; };

template <typename T>
struct image {
    image() = default;
    ~image() = default;

    void resize(int width, int height);
    void allocate(int width, int height);
    void deallocate();

    T* gpu() const;

    int width = 0;
    int height = 0;
    T* data = nullptr;
};


template <typename T>
void image<T>::resize(int width, int height)
{
    if (this->width == width && this->height == height) return;
    deallocate();
    allocate(width, height);
}


template <typename T>
void image<T>::allocate(int width, int height)
{
    this->width = width;
    this->height = height;
    cudaHostAlloc((void**)(&data), width * height * sizeof(T), cudaHostAllocMapped);
}


template <typename T>
void image<T>::deallocate()
{
    cudaFreeHost(this->data);
    this->width = 0;
    this->height = 0;
    this->data = nullptr;
}


template <typename T>
T* image<T>::gpu() const
{
    T* ptr = nullptr;
    auto error = cudaHostGetDevicePointer(&ptr, this->data, 0);
    return ptr;
}

}
