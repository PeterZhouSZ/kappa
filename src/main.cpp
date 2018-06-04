#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kinfu/pipeline.hpp>


volume<sdf32f_t> vol;
camera cam{"/media/sutd/storage/scenenn/061/061.oni"};

constexpr int num_levels = 3;
int frame = 0;
int icp_num_iterations = 10;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float bilateral_d_sigma = 0.1f;
float bilateral_r_sigma = 4.0f;
float cutoff = 4.0f;
float near = 0.001f;
float far = 4.0f;
float mu = 0.1f;
mat4x4 P;
std::vector<mat4x4> poses;

image<rgb8_t>   im;
image<uint16_t> rm;
image<float>    dm;
image<rgb8_t>   cm;
image<float>    dmaps[num_levels];
image<float3>   vmaps[num_levels];
image<float3>   nmaps[num_levels];
image<float3>   rvmaps[num_levels];
image<float3>   rnmaps[num_levels];


static void preallocate()
{
    im.resize(cam.K.width, cam.K.height, ALLOCATOR_MAPPED);
    dm.resize(cam.K.width, cam.K.height, ALLOCATOR_DEVICE);
    for (int level = 0; level < num_levels; ++level) {
        int width = cam.K.width >> level;
        int height = cam.K.height >> level;
        dmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        vmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        nmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        rvmaps[level].resize(width, height, ALLOCATOR_DEVICE);
        rnmaps[level].resize(width, height, ALLOCATOR_DEVICE);
    }
}

static void preprocess()
{
    compute_depth_map(&rm, &dm, cam.K, cutoff);
    bilateral_filter(&dm, &dmaps[0], cam.K, bilateral_d_sigma, bilateral_r_sigma);
    compute_vertex_map(&dmaps[0], &vmaps[0], cam.K);
    compute_normal_map(&vmaps[0], &nmaps[0], cam.K);
}

static void track()
{
    P = icp_p2p_se3(&vmaps[0], &nmaps[0], &rvmaps[0], &rnmaps[0], cam.K, P, icp_num_iterations, dist_threshold, angle_threshold);
}


int main(int argc, char** argv)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        exit(1);
    }
    GLFWwindow* win = glfwCreateWindow(640, 480, "kinfu", NULL, NULL);
    glfwMakeContextCurrent(win);

    // camera cam;
    cam.set_resolution(RESOLUTION_VGA);
    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;
    cam.start();

    int3 dimension = {512, 512, 512};
    vol.voxel_size = 0.008f;
    vol.offset = {-2.0f, -2.0f, 0.0f};
    vol.allocate(dimension, ALLOCATOR_DEVICE);

    preallocate();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 640, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 640, 480, 0, -1 , 1);

        cam.read(&rm, &cm);
        preprocess();
        if (frame > 0) track();
        integrate_volume(&vol, &dm, cam.K, P.inverse(), mu);
        raycast_volume(&vol, &rvmaps[0], &rnmaps[0], cam.K, P, near, far);
        render_phong_light(&im, &rvmaps[0], &rnmaps[0], cam.K);
        frame++;

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
