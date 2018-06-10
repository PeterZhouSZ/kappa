#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kappa/core.hpp>


constexpr int num_levels = 3;
int frame = 0;
int icp_num_iterations = 10;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float bilateral_d_sigma = 0.1f;
float bilateral_r_sigma = 4.0f;
float max_weight = 100.0f;
float cutoff = 4.0f;
float near = 0.001f;
float far = 4.0f;
float mu = 0.1f;

image<rgb8_t>   rm;
image<uint16_t> rdm;
image<float>    dm;
image<rgb8_t>   cm;
image<uint32_t> im;
image<float>    dm0[num_levels];
image<float3>   vm0[num_levels];
image<float3>   nm0[num_levels];
image<float3>   vm1[num_levels];
image<float3>   nm1[num_levels];

cloud<surfel32f_t> pc;
camera cam{"/media/sutd/storage/scenenn/061/061.oni"};
mat4x4 P;


static void preallocate()
{
    rm.resize(cam.K.width, cam.K.height, DEVICE_CUDA_MAPPED);
    dm.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    im.resize(cam.K.width, cam.K.height, DEVICE_CUDA);
    for (int level = 0; level < num_levels; ++level) {
        int width = cam.K.width >> level;
        int height = cam.K.height >> level;
        dm0[level].resize(width, height, DEVICE_CUDA);
        vm0[level].resize(width, height, DEVICE_CUDA);
        nm0[level].resize(width, height, DEVICE_CUDA);
        vm1[level].resize(width, height, DEVICE_CUDA);
        nm1[level].resize(width, height, DEVICE_CUDA);
    }
}

static void preprocess()
{
    compute_depth_map(&rdm, &dm, cam.K, cutoff);
    depth_bilateral(&dm, &dm0[0], cam.K, bilateral_d_sigma, bilateral_r_sigma);
    compute_vertex_map(&dm0[0], &vm0[0], cam.K);
    compute_normal_map(&vm0[0], &nm0[0], cam.K);
}

static void track()
{
    P = icp_p2p_se3(&vm0[0], &nm0[0], &vm1[0], &nm1[0], cam.K, P, icp_num_iterations, dist_threshold, angle_threshold);
}


int main(int argc, char** argv)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        exit(1);
    }
    GLFWwindow* win = glfwCreateWindow(640, 480, "demo", NULL, NULL);
    glfwMakeContextCurrent(win);

    cam.set_resolution(RESOLUTION_VGA);
    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;
    cam.start();

    int size = 0x1000000;
    pc.allocate(size, DEVICE_CUDA);

    preallocate();

    im.clear();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 640, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 640, 480, 0, -1 , 1);

        cam.read(&rdm, &cm);
        preprocess();
        integrate_cloud(&pc, &dm, &im, cam.K, P, mu, max_weight);
        frame++;

        glfwSwapBuffers(win);
        if (frame == 1) glfwSetWindowShouldClose(win, GL_TRUE);
    }

    return 0;
}
