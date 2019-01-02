#include <kappa/camera.hpp>
#include <stdio.h>


using namespace openni;

camera::camera(const char* uri)
{
    status = OpenNI::initialize();
    status = device.open(uri);
    openni::PlaybackControl* control = device.getPlaybackControl();
    if (control) control->setSpeed(-1);
    status = depth.create(device, SENSOR_DEPTH);
    status = color.create(device, SENSOR_COLOR);
    if (depth.isPropertySupported(STREAM_PROPERTY_MIRRORING))
        status = depth.setMirroringEnabled(false);
    if (color.isPropertySupported(STREAM_PROPERTY_MIRRORING))
        status = color.setMirroringEnabled(false);
    status = device.setDepthColorSyncEnabled(true);
    assert(status == STATUS_OK);
}

camera::~camera()
{
    depth.stop();
    color.stop();
    depth.destroy();
    color.destroy();
    device.close();
    OpenNI::shutdown();
}

void camera::start()
{
    status = depth.start();
    status = color.start();
    assert(status == STATUS_OK);
}

bool camera::read(image<uint16_t>* dm)
{
    status = depth.readFrame(&frame);
    int width = frame.getWidth();
    int height = frame.getHeight();
    const void* data = frame.getData();
    size_t size = frame.getDataSize();

    dm->resize(width, height, DEVICE_CUDA_MAPPED);
    memcpy(dm->data, data, size);
    return status == STATUS_OK;
}

bool camera::read(image<rgb8>* cm)
{
    status = color.readFrame(&frame);
    int width = frame.getWidth();
    int height = frame.getHeight();
    const void* data = frame.getData();
    size_t size = frame.getDataSize();

    cm->resize(width, height, DEVICE_CUDA_MAPPED);
    memcpy(cm->data, data, size);
    return status == STATUS_OK;
}

void camera::resolution(int stream, int res)
{
    int width;
    int height;
    const int fps = 30;
    VideoMode mode;

    switch (res) {
        case RESOLUTION_QVGA:
            width = 320;
            height = 240;
            break;
        case RESOLUTION_VGA:
            width = 640;
            height = 480;
            break;
        default:
            break;
    };

    switch (stream) {
        case STREAM_DEPTH:
            mode = depth.getVideoMode();
            mode.setResolution(width, height);
            mode.setFps(fps);
            depth.setVideoMode(mode);
            break;
        case STREAM_COLOR:
            mode = color.getVideoMode();
            mode.setResolution(width, height);
            mode.setFps(fps);
            color.setVideoMode(mode);
            break;
    };
}
