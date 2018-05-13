#pragma once


namespace kinfu {

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}


struct mat4x4 {
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

};
