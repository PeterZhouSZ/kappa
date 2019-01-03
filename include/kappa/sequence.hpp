#pragma once
#include "image.hpp"


struct sequence {
    sequence(const char* path);

    void read(image<uint16_t>* dm);
    void read(image<rgb8>* cm);

    intrinsics K;
    int frame;
    int size;
    float factor;
};
