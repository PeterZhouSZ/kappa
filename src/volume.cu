#include <kappa/volume.hpp>
#include <kappa/math.hpp>


__device__
float tsdf_at(volume<sdf32f_t> vol, int x, int y, int z)
{
    int i = x + y * vol.dimension.x + z * vol.dimension.x * vol.dimension.y;
    if (x < 0 || x >= vol.dimension.x ||
        y < 0 || y >= vol.dimension.y ||
        z < 0 || z >= vol.dimension.z)
        return 1.0f; // cannot interpolate
    return vol[i].tsdf;
}


__device__
float nearest_tsdf(volume<sdf32f_t> vol, float3 p)
{
    int x = roundf((p.x - vol.offset.x) / vol.voxel_size);
    int y = roundf((p.y - vol.offset.y) / vol.voxel_size);
    int z = roundf((p.z - vol.offset.z) / vol.voxel_size);
    return tsdf_at(vol, x, y, z);
}


__device__
float interp_tsdf(volume<sdf32f_t> vol, float3 p)
{
    float3 q = (p - vol.offset) / vol.voxel_size;
    int x = (int)q.x;
    int y = (int)q.y;
    int z = (int)q.z;
    float a = q.x - x;
    float b = q.y - y;
    float c = q.z - z;

    float tsdf = 0.0f;
    tsdf += tsdf_at(vol, x + 0, y + 0, z + 0) * (1 - a) * (1 - b) * (1 - c);
    tsdf += tsdf_at(vol, x + 0, y + 0, z + 1) * (1 - a) * (1 - b) * (    c);
    tsdf += tsdf_at(vol, x + 0, y + 1, z + 0) * (1 - a) * (    b) * (1 - c);
    tsdf += tsdf_at(vol, x + 0, y + 1, z + 1) * (1 - a) * (    b) * (    c);
    tsdf += tsdf_at(vol, x + 1, y + 0, z + 0) * (    a) * (1 - b) * (1 - c);
    tsdf += tsdf_at(vol, x + 1, y + 0, z + 1) * (    a) * (1 - b) * (    c);
    tsdf += tsdf_at(vol, x + 1, y + 1, z + 0) * (    a) * (    b) * (1 - c);
    tsdf += tsdf_at(vol, x + 1, y + 1, z + 1) * (    a) * (    b) * (    c);
    return tsdf;
}


__device__
float3 grad_tsdf(volume<sdf32f_t> vol, float3 p)
{
    int x = roundf((p.x - vol.offset.x) / vol.voxel_size);
    int y = roundf((p.y - vol.offset.y) / vol.voxel_size);
    int z = roundf((p.z - vol.offset.z) / vol.voxel_size);

    float3 grad;
    float f0, f1;
    f0 = tsdf_at(vol, x - 1, y, z);
    f1 = tsdf_at(vol, x + 1, y, z);
    grad.x = (f1 - f0) / vol.voxel_size;
    f0 = tsdf_at(vol, x, y - 1, z);
    f1 = tsdf_at(vol, x, y + 1, z);
    grad.y = (f1 - f0) / vol.voxel_size;
    f0 = tsdf_at(vol, x, y, z - 1);
    f1 = tsdf_at(vol, x, y, z + 1);
    grad.z = (f1 - f0) / vol.voxel_size;
    if (length(grad) == 0.0f) return {0.0f, 0.0f, 0.0f};
    return normalize(grad);
}
