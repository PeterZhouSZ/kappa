#pragma once
#include <openni2/OpenNI.h>
#include "image.hpp"


struct camera {
    camera(const char* uri = openni::ANY_DEVICE);
    ~camera();

    void start();
    bool read(image<uint16_t>* dm);
    bool read(image<rgb8>* cm);
    void resolution(int stream, int res);

    intrinsics K;
    openni::Status status;
    openni::Device device;
    openni::VideoStream depth;
    openni::VideoStream color;
    openni::VideoFrameRef frame;
};
