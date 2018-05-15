#pragma once
#include <openni2/OpenNI.h>
#include "image.hpp"


enum distortion {
    DISTORTION_NONE,
    DISTORTION_FTHETA,
};


enum resolution {
    RESOLUTION_QVGA,
    RESOLUTION_VGA,
};


struct intrinsics {
    int width;
    int height;
    float cx;
    float cy;
    float fx;
    float fy;
    distortion model;
    float coeffs[5];
};


struct camera {
    camera(const char* uri = openni::ANY_DEVICE);
    ~camera();

    void start();
    bool read(image<uint16_t>* dm, image<rgb8_t>* cm);
    void set_resolution(resolution res);

    intrinsics K;
    openni::Status status;
    openni::Device device;
    openni::VideoStream depth;
    openni::VideoStream color;
};
