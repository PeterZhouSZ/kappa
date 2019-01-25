#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kappa/core.hpp>


int frame = 0;
int num_iterations = 10;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float d_sigma = 0.1f;
float r_sigma = 4.0f;
float maxw = 100.0f;
float cutoff = 4.0f;
float near = 0.001f;
float far = 4.0f;
float mu = 0.1f;
float factor = 0.001f;
float3 light = {0.0f, 0.0f, 0.0f};

image<rgb8>     im;
image<uint16_t> rdm;
image<rgb8>     rcm;
image<float>    dm;
image<float>    dm0;
image<float3>   vm0;
image<float4>   nm0;
image<float3>   cm0;
image<float3>   vm1;
image<float4>   nm1;
image<float3>   cm1;

volume<voxel> vol;
camera cam{"/run/media/hieu/storage/scenenn/061/061.oni"};
mat4x4 P;


static void prealloc()
{
    im.resize(cam.K.width, cam.K.height, DEVICE_CUDA_MAPPED);
    dm.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    dm0.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    vm0.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    nm0.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    cm0.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    vm1.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    nm1.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
}


int main(int argc, char** argv)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        exit(1);
    }
    GLFWwindow* win = glfwCreateWindow(640, 480, "demo", nullptr, nullptr);
    glfwMakeContextCurrent(win);

    cam.resolution(STREAM_DEPTH, RESOLUTION_VGA);
    cam.resolution(STREAM_COLOR, RESOLUTION_VGA);
    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;
    cam.start();

    int3 shape = {512, 512, 512};
    vol.voxel_size = 0.008f;
    vol.offset = {-2.0f, -2.0f, 0.0f};
    vol.alloc(shape, DEVICE_CUDA);
    reset_volume(&vol);

    prealloc();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 640, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 640, 480, 0, -1 , 1);

        cam.read(&rdm);
        cam.read(&rcm);
        raw_to_depth(rdm, &dm, cam.K, factor, cutoff);
        raw_to_color(rcm, &cm0, cam.K);
        depth_bilateral(dm, &dm0, cam.K, d_sigma, r_sigma);
        depth_to_vertex(dm0, &vm0, cam.K);
        vertex_to_normal(vm0, &nm0, cam.K);

        if (frame > 0)
            P = icp_p2p_se3(vm0, nm0, vm1, nm1, cam.K, P, num_iterations,
                            dist_threshold, angle_threshold);

        integrate_volume(&vol, dm, cm0, cam.K, P, mu, maxw);
        raycast_volume(vol, &vm1, &nm1, cam.K, P, mu, near, far);

        float3 light = {P.m03, P.m13, P.m23};
        float3 view = {P.m03, P.m13, P.m23};
        render_phong_light(vm1, nm1, &im, cam.K, light, view);
        frame++;

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
