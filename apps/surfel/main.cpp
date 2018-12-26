#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kappa/core.hpp>


constexpr int levels = 3;
int frame = 0;
int num_iterations = 10;
int period = 20;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float d_sigma = 0.1f;
float r_sigma = 4.0f;
float cutoff = 4.0f;
float near = 0.001f;
float far = 4.0f;
float delta_r = 1.5f;
float maxw = 10.0f;
float3 light = {0.0f, 0.0f, 0.0f};

image<rgb8>     im;
image<uint16_t> rdm;
image<float>    dm;
image<rgb8>     cm;
image<uint32_t> idm;
image<float>    dm0[levels];
image<float3>   vm0[levels];
image<float3>   vm1[levels];
image<float4>   nm0[levels];
image<float4>   nm1[levels];

cloud<surfel> pcd;
camera cam{"/run/media/hieu/storage/scenenn/061/061.oni"};
mat4x4 P;


static void prealloc()
{
    im.resize(cam.K.width, cam.K.height, DEVICE_CUDA_MAPPED);
    dm.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    idm.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    for (int level = 0; level < levels; ++level) {
        int width = cam.K.width >> level;
        int height = cam.K.height >> level;
        dm0[level].resize(width, height, DEVICE_CUDA);
        vm0[level].resize(width, height, DEVICE_CUDA);
        nm0[level].resize(width, height, DEVICE_CUDA);
        vm1[level].resize(width, height, DEVICE_CUDA);
        nm1[level].resize(width, height, DEVICE_CUDA);
    }
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

    int size = 0x1000000;
    pcd.alloc(size, DEVICE_CUDA);

    prealloc();
    idm.clear();
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
        cam.read(&cm);
        raw_to_depth(rdm, &dm, cam.K, cutoff);
        depth_bilateral(dm, &dm0[0], cam.K, d_sigma, r_sigma);
        depth_to_vertex(dm0[0], &vm0[0], cam.K);
        vertex_to_normal_radius(vm0[0], &nm0[0], cam.K);

        if (frame > 0)
            P = icp_p2p_se3(vm0[0], nm0[0], vm1[0], nm1[0], cam.K, P,
                num_iterations, dist_threshold, angle_threshold);

        integrate(&pcd, vm0[0], nm0[0], idm, cam.K, P, frame, delta_r);
        cleanup(&pcd, maxw, frame, period);
        raycast(pcd, &vm1[0], &nm1[0], &idm, cam.K, P, maxw);
        float3 view = {P.m03, P.m13, P.m23};
        render_phong_light(vm1[0], nm1[0], &im, cam.K, light, view);
        frame++;

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
