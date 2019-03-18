#pragma once
#include "common.hpp"
#include "image.hpp"
#include <stdio.h>


struct sequence {
    sequence(const char* path);
    ~sequence();

    void start();
    bool end() const;

    bool read(image<uint16_t>* dm);
    bool read(image<rgb8>* cm);
    bool read(mat4x4* P);

    char path[256];
    FILE* depth = nullptr;
    FILE* color = nullptr;
    FILE* pose  = nullptr;
};
