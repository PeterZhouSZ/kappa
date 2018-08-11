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
    if (depth.isPropertySupported(openni::STREAM_PROPERTY_MIRRORING))
        status = depth.setMirroringEnabled(false);
    if (color.isPropertySupported(openni::STREAM_PROPERTY_MIRRORING))
        status = color.setMirroringEnabled(false);
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


bool camera::read(image<uint16_t>* dm, image<rgb8_t>* cm)
{
    if (dm) {
        VideoFrameRef frame;
        status = depth.readFrame(&frame);
        dm->resize(frame.getWidth(), frame.getHeight(), DEVICE_CUDA_MAPPED);
        memcpy(dm->data, frame.getData(), frame.getDataSize());
    }
    if (cm) {
        VideoFrameRef frame;
        status = color.readFrame(&frame);
        cm->resize(frame.getWidth(), frame.getHeight(), DEVICE_CUDA_MAPPED);
        memcpy(cm->data, frame.getData(), frame.getDataSize());
    }
    return status == STATUS_OK;
}


void camera::set_resolution(resolution res)
{
    int width;
    int height;
    const int fps = 30;

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

    VideoMode mode;
    mode = depth.getVideoMode();
    mode.setResolution(width, height);
    mode.setFps(fps);
    depth.setVideoMode(mode);
    mode = color.getVideoMode();
    mode.setResolution(width, height);
    mode.setFps(fps);
    color.setVideoMode(mode);
}
