#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>
#include <kinfu/pipeline.hpp>
#include <kinfu/renderer.hpp>


int main(int argc, char** argv)
{
    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        exit(1);
    }
    GLFWwindow* win = glfwCreateWindow(1280, 480, "kinfu", NULL, NULL);
    glfwMakeContextCurrent(win);

    // camera cam;
    camera cam{"/media/sutd/storage/scenenn/061/061.oni"};
    cam.set_resolution(RESOLUTION_VGA);
    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;
    cam.start();

    volume<sdf32f_t> vol;
    int3 dimension = {512, 512, 512};
    vol.voxel_size = 0.008f;
    vol.offset = {-2.0f, -2.0f, 0.0f};
    vol.allocate(dimension, ALLOCATOR_DEVICE);

    pipeline pipe;
    pipe.cam = &cam;
    pipe.vol = &vol;

    image<rgb8_t> im;
    renderer r;
    r.K = cam.K;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 1280, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 1280, 480, 0, -1 , 1);

        pipe.process();
        r.render_phong(&im, &pipe.rvmaps[0], &pipe.rnmaps[0]);

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(im.width, im.height, GL_RGB, GL_UNSIGNED_BYTE, im.data);
        glfwSwapBuffers(win);
    }

    return 0;
}
