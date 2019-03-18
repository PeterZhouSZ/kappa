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
    int channels = png_get_channels(pp, pi);
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
    sprintf(fname, "%s/depth.txt", path);
    depth = fopen(fname, "r");
    sprintf(fname, "%s/rgb.txt", path);
    color = fopen(fname, "r");
    sprintf(fname, "%s/pose.txt", path);
    pose = fopen(fname, "r");
}


sequence::~sequence()
{
    fclose(depth);
    fclose(color);
    fclose(pose);
}


void sequence::start()
{
}


bool sequence::end() const
{
    return feof(depth) || feof(color);
}


bool sequence::read(image<uint16_t>* dm)
{
    char str[256];
    if (fgets(str, 256, depth)) {
        char fname[512];
        str[strcspn(str, "\n")] = '\0'; // remove trailing newline
        sprintf(fname, "%s/%s", path, str);
        read_image_png(fname, dm);
        return true;
    }
    return false;
}


bool sequence::read(image<rgb8>* cm)
{
    char str[256];
    if (fgets(str, 256, color)) {
        char fname[512];
        str[strcspn(str, "\n")] = '\0'; // remove trailing newline
        sprintf(fname, "%s/%s", path, str);
        read_image_png(fname, cm);
        return true;
    }
    return false;
}


bool sequence::read(mat4x4* P)
{
    char str[256];
    if (fgets(str, 256, pose)) {
        char fname[512];
        str[strcspn(str, "\n")] = '\0'; // remove trailing newline
        sprintf(fname, "%s/%s", path, str);
        FILE* fp = fopen(fname, "r");
        fscanf(fp, "%f %f %f %f\n"
               "%f %f %f %f\n"
               "%f %f %f %f\n"
               "%f %f %f %f\n",
               &(P->m00), &(P->m01), &(P->m02), &(P->m03),
               &(P->m10), &(P->m11), &(P->m12), &(P->m13),
               &(P->m20), &(P->m21), &(P->m22), &(P->m23),
               &(P->m30), &(P->m31), &(P->m32), &(P->m33));
        fclose(fp);
        return true;
    }
    return false;
}
