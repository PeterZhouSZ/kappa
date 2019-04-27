#include <kappa/sequence.hpp>
#include <string.h>
#include <png.h>


template <typename T>
static void read_image_png(const char* fname, image<T>* im)
{
    FILE* fp = fopen(fname, "rb");
    png_structp pp = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop pi = png_create_info_struct(pp);
    if (!pp || !pi) return;
    if (setjmp(png_jmpbuf(pp))) return;

    png_init_io(pp, fp);
    png_read_info(pp, pi);

    int width    = png_get_image_width(pp, pi);
    int height   = png_get_image_height(pp, pi);
    int bitdepth = png_get_bit_depth(pp, pi);

    if (bitdepth == 16) png_set_swap(pp);

    im->resize(width, height, DEVICE_CUDA_MAPPED);
    png_bytep* rows = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; ++i)
        rows[i] = (png_bytep)(im->data + i * width);

    png_read_image(pp, rows);
    free(rows);
    fclose(fp);
}


sequence::sequence(const char* path)
{
    char fname[512];
    strcpy(this->path, path);
    sprintf(fname, "%s/trajectory.log", path);
    fp = fopen(fname, "r");
}


sequence::~sequence()
{
    fclose(fp);
}


bool sequence::end() const
{
    return feof(fp);
}


bool sequence::read(image<uint16_t>* dm)
{
    char fname[512];
    sprintf(fname, "%s/depth/%06d.png", path, frame);
    read_image_png(fname, dm);
    return true;
}


bool sequence::read(image<rgb8>* cm)
{
    char fname[512];
    sprintf(fname, "%s/color/%06d.png", path, frame);
    read_image_png(fname, cm);
    return true;
}


bool sequence::read(mat4x4* P)
{
    fscanf(fp, "%*d %*d %*d\n"
           "%f %f %f %f\n"
           "%f %f %f %f\n"
           "%f %f %f %f\n"
           "%f %f %f %f\n",
           &(P->m00), &(P->m01), &(P->m02), &(P->m03),
           &(P->m10), &(P->m11), &(P->m12), &(P->m13),
           &(P->m20), &(P->m21), &(P->m22), &(P->m23),
           &(P->m30), &(P->m31), &(P->m32), &(P->m33));
    return true;
}


void sequence::seek(int start)
{
    mat4x4 P;
    while (frame < start) {
        read(&P);
        frame++;
    }
}


void sequence::next()
{
    frame++;
}
