#pragma once
#include "common.hpp"


template <typename T>
struct volume {
    volume() = default;
    ~volume() = default;

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

    GPU_CODE
    float operator()(int x, int y, int z)
    {
        int i = x + y * shape.x + z * shape.x * shape.y;
        if (x < 0 || x >= shape.x ||
            y < 0 || y >= shape.y ||
            z < 0 || z >= shape.z)
            return 1.0f; // cannot interpolate
        return data[i].tsdf;
    }

    GPU_CODE
    float nearest(float3 p)
    {
        int x = roundf((p.x - offset.x) / voxel_size);
        int y = roundf((p.y - offset.y) / voxel_size);
        int z = roundf((p.z - offset.z) / voxel_size);
        return (*this)(x, y, z);
    }


    GPU_CODE
    float interp(float3 p)
    {
        float3 q = (p - offset) / voxel_size;
        int x = (int)q.x;
        int y = (int)q.y;
        int z = (int)q.z;
        float a = q.x - x;
        float b = q.y - y;
        float c = q.z - z;

        float tsdf = 0.0f;
        tsdf += (*this)(x + 0, y + 0, z + 0) * (1 - a) * (1 - b) * (1 - c);
        tsdf += (*this)(x + 0, y + 0, z + 1) * (1 - a) * (1 - b) * (    c);
        tsdf += (*this)(x + 0, y + 1, z + 0) * (1 - a) * (    b) * (1 - c);
        tsdf += (*this)(x + 0, y + 1, z + 1) * (1 - a) * (    b) * (    c);
        tsdf += (*this)(x + 1, y + 0, z + 0) * (    a) * (1 - b) * (1 - c);
        tsdf += (*this)(x + 1, y + 0, z + 1) * (    a) * (1 - b) * (    c);
        tsdf += (*this)(x + 1, y + 1, z + 0) * (    a) * (    b) * (1 - c);
        tsdf += (*this)(x + 1, y + 1, z + 1) * (    a) * (    b) * (    c);
        return tsdf;
    }

    GPU_CODE
    float3 color(float3 p)
    {
        int x = roundf((p.x - offset.x) / voxel_size);
        int y = roundf((p.y - offset.y) / voxel_size);
        int z = roundf((p.z - offset.z) / voxel_size);
        int i = x + y * shape.x + z * shape.x * shape.y;
        if (x < 0 || x >= shape.x ||
            y < 0 || y >= shape.y ||
            z < 0 || z >= shape.z)
            return {0.0f, 0.0f, 0.0f}; // cannot interpolate
        return data[i].color;
    }

    GPU_CODE
    float3 grad(float3 p)
    {
        int x = roundf((p.x - offset.x) / voxel_size);
        int y = roundf((p.y - offset.y) / voxel_size);
        int z = roundf((p.z - offset.z) / voxel_size);

        float3 delta;
        delta.x = (*this)(x + 1, y, z) - (*this)(x - 1, y, z);
        delta.y = (*this)(x, y + 1, z) - (*this)(x, y - 1, z);
        delta.z = (*this)(x, y, z + 1) - (*this)(x, y, z - 1);
        delta = delta / voxel_size;
        if (length(delta) == 0.0f)
            return {0.0f, 0.0f, 0.0f};
        return normalize(delta);
    }

    void alloc(int3 shape, int device);
    void free();

    volume<T> cpu() const;
    volume<T> cuda() const;

    int3 shape = {0, 0, 0};
    float3 offset = {0.0f, 0.0f, 0.0f};
    float voxel_size = 0.0f;
    int device;
    T* data = nullptr;
};


template <typename T>
void volume<T>::alloc(int3 shape, int device)
{
    this->shape = shape;
    this->device = device;
    size_t size = shape.x * shape.y * shape.z * sizeof(T);
    switch (device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            cudaMalloc((void**)&data, size);
        case DEVICE_CUDA_MAPPED:
            break;
    }
}


template <typename T>
void volume<T>::free()
{
    switch (this->device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            cudaFree(data);
            break;
        case DEVICE_CUDA_MAPPED:
            break;
    };
    data = nullptr;
    shape = {0, 0, 0};
}


template <typename T>
volume<T> volume<T>::cuda() const
{
    volume<T> vol;
    vol.shape = shape;
    vol.offset = offset;
    vol.voxel_size = voxel_size;
    vol.device = DEVICE_CUDA;
    switch (this->device) {
        case DEVICE_CPU:
            break;
        case DEVICE_CUDA:
            vol.data = data;
            break;
        case DEVICE_CUDA_MAPPED:
            break;
    }
    return vol;
}
