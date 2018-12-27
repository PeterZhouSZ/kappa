#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <Eigen/Core>
#include <Eigen/LU>

#ifdef __NVCC__
#define GPU_CODE __device__
#define CPU_GPU_CODE __host__ __device__
#else
#define GPU_CODE
#define CPU_GPU_CODE
#endif

#define ZBUFFER_SCALE 100000
#define Z_OFFSET 10000000

enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_CUDA_MAPPED,
};

enum {
    DISTORTION_NONE,
    DISTORTION_FTHETA,
};

enum {
    RESOLUTION_QVGA,
    RESOLUTION_VGA,
};

enum {
    STREAM_DEPTH,
    STREAM_COLOR,
};


struct rgb8 { uint8_t r, g, b; };

struct voxel {
    float tsdf;
    float weight;
    rgb8  color;
};

struct surfel {
    float3 pos;
    float3 normal;
    float radius;
    float weight;
    int timestamp;
};

struct JtJse3 {
    float weight = 0.0f;
    float error = 0.0f;
    float Jte[6] = {0.0f};
    float JtJ[21] = {0.0f};

    __host__ __device__
    const JtJse3& operator+=(const JtJse3& other)
    {
        error += other.error;
        weight += other.weight;
        for (int i = 0; i < 6; ++i)
            Jte[i] += other.Jte[i];
        for (int i = 0; i < 21; ++i)
            JtJ[i] += other.JtJ[i];
        return *this;
    }
};

struct mat4x4 {
    mat4x4()
    {
        m00 = 1.0f; m01 = 0.0f; m02 = 0.0f; m03 = 0.0f;
        m10 = 0.0f; m11 = 1.0f; m12 = 0.0f; m13 = 0.0f;
        m20 = 0.0f; m21 = 0.0f; m22 = 1.0f; m23 = 0.0f;
        m30 = 0.0f; m31 = 0.0f; m32 = 0.0f; m33 = 1.0f;
    }

    mat4x4(const Eigen::Matrix4f& m)
    {
        m00 = m(0, 0); m01 = m(0, 1); m02 = m(0, 2); m03 = m(0, 3);
        m10 = m(1, 0); m11 = m(1, 1); m12 = m(1, 2); m13 = m(1, 3);
        m20 = m(2, 0); m21 = m(2, 1); m22 = m(2, 2); m23 = m(2, 3);
        m30 = m(3, 0); m31 = m(3, 1); m32 = m(3, 2); m33 = m(3, 3);
    }

    mat4x4(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
    {
        m00 = R(0, 0); m01 = R(0, 1); m02 = R(0, 2); m03 = t(0, 3);
        m10 = R(1, 0); m11 = R(1, 1); m12 = R(1, 2); m13 = t(1, 3);
        m20 = R(2, 0); m21 = R(2, 1); m22 = R(2, 2); m23 = t(2, 3);
        m30 = 0.0f;    m31 = 0.0f;    m32 = 0.0f;    m33 = 1.0f;
    }

    mat4x4 inverse() const
    {
        Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>
            m((float*)data);
        Eigen::Matrix4f i = m.inverse();
        return mat4x4(i);
    }

    union {
        struct {
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
            float m30, m31, m32, m33;
        };
        float data[16];
    };
};

inline mat4x4 operator*(mat4x4 A, mat4x4 B)
{
    mat4x4 C;
    C.m00 = A.m00 * B.m00 + A.m01 * B.m10 + A.m02 * B.m20 + A.m03 * B.m30;
    C.m01 = A.m00 * B.m01 + A.m01 * B.m11 + A.m02 * B.m21 + A.m03 * B.m31;
    C.m02 = A.m00 * B.m02 + A.m01 * B.m12 + A.m02 * B.m22 + A.m03 * B.m32;
    C.m03 = A.m00 * B.m03 + A.m01 * B.m13 + A.m02 * B.m23 + A.m03 * B.m33;

    C.m10 = A.m10 * B.m00 + A.m11 * B.m10 + A.m12 * B.m20 + A.m13 * B.m30;
    C.m11 = A.m10 * B.m01 + A.m11 * B.m11 + A.m12 * B.m21 + A.m13 * B.m31;
    C.m12 = A.m10 * B.m02 + A.m11 * B.m12 + A.m12 * B.m22 + A.m13 * B.m32;
    C.m13 = A.m10 * B.m03 + A.m11 * B.m13 + A.m12 * B.m23 + A.m13 * B.m33;

    C.m20 = A.m20 * B.m00 + A.m21 * B.m10 + A.m22 * B.m20 + A.m23 * B.m30;
    C.m21 = A.m20 * B.m01 + A.m21 * B.m11 + A.m22 * B.m21 + A.m23 * B.m31;
    C.m22 = A.m20 * B.m02 + A.m21 * B.m12 + A.m22 * B.m22 + A.m23 * B.m32;
    C.m23 = A.m20 * B.m03 + A.m21 * B.m13 + A.m22 * B.m23 + A.m23 * B.m33;

    C.m30 = A.m30 * B.m00 + A.m31 * B.m10 + A.m32 * B.m20 + A.m33 * B.m30;
    C.m31 = A.m30 * B.m01 + A.m31 * B.m11 + A.m32 * B.m21 + A.m33 * B.m31;
    C.m32 = A.m30 * B.m02 + A.m31 * B.m12 + A.m32 * B.m22 + A.m33 * B.m32;
    C.m33 = A.m30 * B.m03 + A.m31 * B.m13 + A.m32 * B.m23 + A.m33 * B.m33;
    return C;
}

CPU_GPU_CODE
inline float3 operator*(mat4x4 A, float3 v)
{
    float3 u;
    u.x = A.m00 * v.x + A.m01 * v.y + A.m02 * v.z + A.m03;
    u.y = A.m10 * v.x + A.m11 * v.y + A.m12 * v.z + A.m13;
    u.z = A.m20 * v.x + A.m21 * v.y + A.m22 * v.z + A.m23;
    return u;
}

CPU_GPU_CODE
inline float3 rotate(mat4x4 A, float3 v)
{
    float3 u;
    u.x = A.m00 * v.x + A.m01 * v.y + A.m02 * v.z;
    u.y = A.m10 * v.x + A.m11 * v.y + A.m12 * v.z;
    u.z = A.m20 * v.x + A.m21 * v.y + A.m22 * v.z;
    return u;
}

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}
