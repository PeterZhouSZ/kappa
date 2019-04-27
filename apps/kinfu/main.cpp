#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kappa/core.hpp>


int num_iterations = 10;
int num_vertices = 0x1000000;
float dist_threshold = 0.05f;
float angle_threshold = 0.8f;
float d_sigma = 0.1f;
float r_sigma = 4.0f;
float maxw = 100.0f;
float cutoff = 4.0f;
float near = 0.001f;
float far = 4.0f;
float mu = 0.04f;
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
array<vertex> va;
mat4x4 P, B;


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
}


static void write_mesh_ply(const char* fname, const array<vertex> va, int size)
{
    FILE* fp = fopen(fname, "wb");
    fprintf(fp, "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex %d\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n", size);
    for (int i = 0; i < size; ++i) {
        uint8_t r = (uint8_t)(va[i].color.x * 255.0f);
        uint8_t g = (uint8_t)(va[i].color.y * 255.0f);
        uint8_t b = (uint8_t)(va[i].color.z * 255.0f);
        fwrite(&va[i].pos.x, sizeof(float), 1, fp);
        fwrite(&va[i].pos.y, sizeof(float), 1, fp);
        fwrite(&va[i].pos.z, sizeof(float), 1, fp);
        fwrite(&r, sizeof(uint8_t), 1, fp);
        fwrite(&g, sizeof(uint8_t), 1, fp);
        fwrite(&b, sizeof(uint8_t), 1, fp);
    }
    fclose(fp);
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

    int start = atoi(argv[2]);
    int end   = atoi(argv[3]);

    sequence seq{argv[1]};
    seq.seek(start);

    intrinsics K;
    K.fx = 577.870605f;
    K.fy = 577.870605f;
    K.cx = 319.5f;
    K.cy = 239.5f;
    K.width = 640;
    K.height = 480;

    int3 shape = {512, 512, 512};
    vol.voxel_size = 0.008f;
    vol.offset = {-2.048f, -2.048f, 0.0f};
    vol.alloc(shape, DEVICE_CUDA);
    reset_volume(&vol);

    prealloc(K);
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 640, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 640, 480, 0, -1 , 1);

        seq.read(&rdm);
        seq.read(&rcm);
        raw_to_depth(rdm, &dm, K, factor, cutoff);
        raw_to_color(rcm, &cm0, K);
        depth_bilateral(dm, &dm0, K, d_sigma, r_sigma);
        depth_to_vertex(dm0, &vm0, K);
        vertex_to_normal(vm0, &nm0, K);

        mat4x4 P;
        seq.read(&P);
        if (seq.frame == start) B = P.inverse();
        P = B * P;
        seq.next();

        integrate_volume(&vol, dm, cm0, K, P, mu, maxw);
        raycast_volume(vol, &vm1, &nm1, &cm1, K, P, mu, near, far);

        float3 light = {P.m03, P.m13, P.m23};
        float3 view = {P.m03, P.m13, P.m23};
        render_phong_light(vm1, nm1, cm1, &im, K, light, view);

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);

        if (seq.frame == end || seq.end()) glfwSetWindowShouldClose(win, true);
    }

    va.alloc(0x1000000, DEVICE_CUDA_MAPPED);
    int size = extract_isosurface_volume(vol, &va);
    write_mesh_ply(argv[4], va, size);

    return 0;
}
