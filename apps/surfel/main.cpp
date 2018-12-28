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
float r_sigma = 3.0f;
float cutoff = 3.0f;
float near = 0.001f;
float far = 4.0f;
float maxw = 10.0f;
float delta_r = 1.5f;

image<rgb8>     im;
image<uint16_t> rdm;
image<float>    dm;
image<rgb8>     cm;
image<uint32_t> idm[levels];
image<float>    dm0[levels];
image<float3>   vm0[levels];
image<float3>   vm1[levels];
image<float4>   nm0[levels];
image<float4>   nm1[levels];

cloud<surfel> pcd;
camera cam{"/run/media/hieu/storage/scenenn/011/011.oni"};
intrinsics K[levels];
mat4x4 P;


static void prealloc()
{
    im.resize(cam.K.width, cam.K.height, DEVICE_CUDA_MAPPED);
    dm.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    for (int level = 0; level < levels; ++level) {
        K[level].width  = cam.K.width  >> level;
        K[level].height = cam.K.height >> level;
        K[level].cx = cam.K.cx * powf(0.5f, level);
        K[level].cy = cam.K.cy * powf(0.5f, level);
        K[level].fx = cam.K.fx * powf(0.5f, level);
        K[level].fy = cam.K.fy * powf(0.5f, level);
        dm0[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        vm0[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        nm0[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        vm1[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        nm1[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        idm[level].resize(K[level].width, K[level].height, DEVICE_CUDA);
        idm[level].clear();
    }
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
    reset(&pcd);

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
        cam.read(&cm);
        raw_to_depth(rdm, &dm, cam.K, cutoff);
        depth_bilateral(dm, &dm0[0], cam.K, d_sigma, r_sigma);
        depth_to_vertex(dm0[0], &vm0[0], cam.K);
        vertex_to_normal_radius(vm0[0], &nm0[0], cam.K);

        if (frame > 0)
            P = icp_p2p_se3(vm0[0], nm0[0], vm1[0], nm1[0], cam.K, P,
                            num_iterations, dist_threshold, angle_threshold);
        // fscanf(fp, "%*d %*d %*d\n"
        //        "%f %f %f %f\n"
        //        "%f %f %f %f\n"
        //        "%f %f %f %f\n"
        //        "%f %f %f %f\n",
        //        &P.m00, &P.m01, &P.m02, &P.m03,
        //        &P.m10, &P.m11, &P.m12, &P.m13,
        //        &P.m20, &P.m21, &P.m22, &P.m23,
        //        &P.m30, &P.m31, &P.m32, &P.m33);

        float3 light = {P.m03, P.m13, P.m23};
        float3 view = {P.m03, P.m13, P.m23};
        integrate(&pcd, vm0[0], nm0[0], idm[0], cam.K, P, frame, delta_r);
        cleanup(&pcd, maxw, frame, period);

        for (int level = 0; level < levels; ++level)
            raycast(pcd, &vm1[level], &nm1[level], &idm[level],
                    K[level], P, frame, maxw, cutoff);
        render_phong_light(vm1[0], nm1[0], &im, K[0], light, view);
        frame++;

        printf("%d %d\n", pcd.size, pcd.capacity);
        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
