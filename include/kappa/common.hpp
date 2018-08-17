#pragma once
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <Eigen/Core>
#include <Eigen/LU>


enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_CUDA_MAPPED,
};


struct rgb8_t { uint8_t r, g, b; };


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
        rows[0] = {1.0f, 0.0f, 0.0f, 0.0f};
        rows[1] = {0.0f, 1.0f, 0.0f, 0.0f};
        rows[2] = {0.0f, 0.0f, 1.0f, 0.0f};
        rows[3] = {0.0f, 0.0f, 0.0f, 1.0f};
    }

    mat4x4(const Eigen::Matrix4f& m)
    {
        rows[0] = {m(0, 0), m(0, 1), m(0, 2), m(0, 3)};
        rows[1] = {m(1, 0), m(1, 1), m(1, 2), m(1, 3)};
        rows[2] = {m(2, 0), m(2, 1), m(2, 2), m(2, 3)};
        rows[3] = {m(3, 0), m(3, 1), m(3, 2), m(3, 3)};
    }

    mat4x4(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
    {
        rows[0] = {R(0, 0), R(0, 1), R(0, 2), t(0)};
        rows[1] = {R(1, 0), R(1, 1), R(1, 2), t(1)};
        rows[2] = {R(2, 0), R(2, 1), R(2, 2), t(2)};
        rows[3] = {   0.0f,    0.0f,    0.0f, 1.0f};
    }

    mat4x4 inverse() const
    {
        Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> m((float*)data);
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
        float4 rows[4];
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


__host__ __device__
inline float3 operator*(mat4x4 A, float3 v)
{
    float3 u;
    u.x = A.m00 * v.x + A.m01 * v.y + A.m02 * v.z + A.m03;
    u.y = A.m10 * v.x + A.m11 * v.y + A.m12 * v.z + A.m13;
    u.z = A.m20 * v.x + A.m21 * v.y + A.m22 * v.z + A.m23;
    return u;
}


__host__ __device__
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


inline void print_mat4x4(FILE* fp, mat4x4 M)
{
    printf("%f %f %f %f\n"
           "%f %f %f %f\n"
           "%f %f %f %f\n"
           "%f %f %f %f\n",
           M.m00, M.m01, M.m02, M.m03,
           M.m10, M.m11, M.m12, M.m13,
           M.m20, M.m21, M.m22, M.m23,
           M.m30, M.m31, M.m32, M.m33);
}
