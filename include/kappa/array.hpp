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
