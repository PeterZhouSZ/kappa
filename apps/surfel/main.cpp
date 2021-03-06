#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kappa/core.hpp>


int frame = 0;
int num_iterations = 10;
int delta_t = 20;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float d_sigma = 0.1f;
float r_sigma = 3.0f;
float factor = 0.001f;
float cutoff = 3.0f;
float near = 0.001f;
float far = 4.0f;
float maxw = 10.0f;
float delta_r = 1.5f;

image<rgb8>     im;
image<uint16_t> rdm;
image<rgb8>     rcm;
image<float>    dm;
image<uint32_t> idm;
image<float>    dm0;
image<float3>   vm0;
image<float4>   nm0;
image<float3>   cm0;
image<float3>   vm1;
image<float4>   nm1;
image<float3>   cm1;

cloud<surfel> pcd;
mat4x4 P;


static void prealloc(intrinsics K)
{
    im.resize(K.width, K.height, DEVICE_CUDA_MAPPED);
    dm.resize(K.width, K.height, DEVICE_CUDA);
    dm0.resize(K.width, K.height, DEVICE_CUDA);
    vm0.resize(K.width, K.height, DEVICE_CUDA);
    nm0.resize(K.width, K.height, DEVICE_CUDA);
    cm0.resize(K.width, K.height, DEVICE_CUDA);
    vm1.resize(K.width, K.height, DEVICE_CUDA);
    nm1.resize(K.width, K.height, DEVICE_CUDA);
    cm1.resize(K.width, K.height, DEVICE_CUDA);
    idm.resize(K.width, K.height, DEVICE_CUDA);
    idm.clear();
}


int main(int argc, char** argv)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        return 0;
    }
    GLFWwindow* win = glfwCreateWindow(640, 480, "demo", nullptr, nullptr);
    glfwMakeContextCurrent(win);

    camera cam{argv[1]};
    cam.resolution(STREAM_DEPTH, RESOLUTION_VGA);
    cam.resolution(STREAM_COLOR, RESOLUTION_VGA);
    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;
    cam.start();

    int capacity = 0x1000000;
    pcd.alloc(capacity, DEVICE_CUDA);
    reset_cloud(&pcd);

    prealloc(cam.K);
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
        vertex_to_normal_radius(vm0, &nm0, cam.K);

        if (frame > 0) {
            P = icp_p2p_se3(
                vm0, nm0, vm1, nm1, cam.K, P,
                num_iterations, dist_threshold, angle_threshold);
        }

        integrate_cloud(&pcd, vm0, nm0, cm0, idm, cam.K, P, frame, delta_r);
        cleanup_cloud(&pcd, maxw, frame, delta_t);
        raycast_cloud(pcd, &vm1, &nm1, &cm1, &idm, cam.K, P, frame, maxw, cutoff);

        float3 light = {P.m03, P.m13, P.m23};
        float3 view = {P.m03, P.m13, P.m23};
        render_phong_light(vm1, nm1, cm1, &im, cam.K, light, view);
        frame++;

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
