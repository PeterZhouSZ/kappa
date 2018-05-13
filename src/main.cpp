#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <GLFW/glfw3.h>

#include <kinfu/kinfu.hpp>


using namespace kinfu;

int main(int argc, char** argv)
{
    if (!glfwInit()) {
        fprintf(stdout, "[GLFW] failed to init!\n");
        exit(1);
    }
    GLFWwindow* win = glfwCreateWindow(640, 480, "kinfu", NULL, NULL);
    glfwMakeContextCurrent(win);

    camera cam;
    cam.set_resolution(RESOLUTION_VGA);
    cam.start();

    cam.K.cx = 320.0f;
    cam.K.cy = 240.0f;
    cam.K.fx = 585.0f;
    cam.K.fy = 585.0f;
    cam.K.width = 640;
    cam.K.height = 480;

    pipeline pipe;
    pipe.register_camera(&cam);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glViewport(0, 0, 640, 480);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 640, 480, 0, -1 , 1);

        pipe.process();

        glPixelZoom(1, -1);
        glRasterPos2i(0, 0);
        glDrawPixels(pipe.vm[0].width, pipe.vm[0].height,
                     GL_RGB, GL_FLOAT, pipe.vm[0].data);
        glfwSwapBuffers(win);
    }

    return 0;
}
