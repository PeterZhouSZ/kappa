#include <kappa/core.hpp>


void write_mesh_ply(const char* fname, const array<vertex> va, int size)
{
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "ply\n"
            "format ascii 1.0\n"
            "element vertex %d\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n", size);
    for (int i = 0; i < size; ++i) {
        uint8_t r = (uint8_t)(va[i].color.x * 255.0f);
        uint8_t g = (uint8_t)(va[i].color.y * 255.0f);
        uint8_t b = (uint8_t)(va[i].color.z * 255.0f);
        fprintf(fp, "%f %f %f ", va[i].pos.x, va[i].pos.y, va[i].pos.z);
        fprintf(fp, "%f %f %f ", va[i].normal.x, va[i].normal.y, va[i].normal.z);
        fprintf(fp, "%d %d %d\n", r, g, b);
    }
    fclose(fp);
}
