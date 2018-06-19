#include <kappa/core.hpp>
#include <float.h>
#include <Eigen/SVD>


__global__
void icp_p2p_se3_kernel(image<JtJse3> JTJ, image<float3> vm0, image<float4> nm0, image<float3> vm1, image<float4> nm1,
                        intrinsics K, mat4x4 T, float dist_threshold, float angle_threshold)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;
    if (u >= K.width || v >= K.height) return;

    int i = u + v * K.width;
    JTJ.data[i].error = 0.0f;
    JTJ.data[i].weight = 0.0f;

    float3 p0 = T * vm0.data[i];
    float3 n0 = rotate(T, make_float3(nm0.data[i]));
    if (p0.z == 0.0f) return;

    float3 p1 = vm1.data[i];
    float3 n1 = make_float3(nm1.data[i]);
    if (p1.z == 0.0f) return;

    float3 d = p1 - p0;
    if (length(d) >= dist_threshold) return;
    if (fabs(dot(n0, n1)) < angle_threshold) return;

    float e = dot(d, n1);
    float3 Jt = n1;
    float3 JR = cross(p0, n1);
    JTJ.data[i].weight  = 1.0f;
    JTJ.data[i].error   = e * e;
    JTJ.data[i].Jte[0]  = e * Jt.x;
    JTJ.data[i].Jte[1]  = e * Jt.y;
    JTJ.data[i].Jte[2]  = e * Jt.z;
    JTJ.data[i].Jte[3]  = e * JR.x;
    JTJ.data[i].Jte[4]  = e * JR.y;
    JTJ.data[i].Jte[5]  = e * JR.z;
    JTJ.data[i].JtJ[0]  = Jt.x * Jt.x;
    JTJ.data[i].JtJ[1]  = Jt.x * Jt.y;
    JTJ.data[i].JtJ[2]  = Jt.x * Jt.z;
    JTJ.data[i].JtJ[3]  = Jt.x * JR.x;
    JTJ.data[i].JtJ[4]  = Jt.x * JR.y;
    JTJ.data[i].JtJ[5]  = Jt.x * JR.z;
    JTJ.data[i].JtJ[6]  = Jt.y * Jt.y;
    JTJ.data[i].JtJ[7]  = Jt.y * Jt.z;
    JTJ.data[i].JtJ[8]  = Jt.y * JR.x;
    JTJ.data[i].JtJ[9]  = Jt.y * JR.y;
    JTJ.data[i].JtJ[10] = Jt.y * JR.z;
    JTJ.data[i].JtJ[11] = Jt.z * Jt.z;
    JTJ.data[i].JtJ[12] = Jt.z * JR.x;
    JTJ.data[i].JtJ[13] = Jt.z * JR.y;
    JTJ.data[i].JtJ[14] = Jt.z * JR.z;
    JTJ.data[i].JtJ[15] = JR.x * JR.x;
    JTJ.data[i].JtJ[16] = JR.x * JR.y;
    JTJ.data[i].JtJ[17] = JR.x * JR.z;
    JTJ.data[i].JtJ[18] = JR.y * JR.y;
    JTJ.data[i].JtJ[19] = JR.y * JR.z;
    JTJ.data[i].JtJ[20] = JR.z * JR.z;
}


__global__
void se3_reduce_kernel(image<JtJse3> JTJ, image<JtJse3> Axb)
{
    extern __shared__ JtJse3 data[];
    int tid = threadIdx.x;

    JtJse3 r;
    for (int v = blockIdx.x; v < JTJ.height; v += gridDim.x) {
        for (int u = threadIdx.x; u < JTJ.width; u += blockDim.x) {
            int i = u + v * JTJ.width;
            if (JTJ.data[i].weight == 0) continue;
            r += JTJ.data[i];
        }
    }
    data[tid] = r;
    __syncthreads();

    if (tid > 0) return;
    for (int i = 1; i < blockDim.x; ++i)
        data[0] += data[i];
    Axb.data[blockIdx.x] = data[0];
}


static Eigen::Matrix3f rodrigues(float3 w)
{
    float theta = length(w);
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f R = I;

    if (theta >= FLT_EPSILON) {
        float c = cos(theta);
        float s = sin(theta);
        float inv_theta = theta != 0 ? 1.0f / theta : 0.0f;

        Eigen::Matrix3f G;
        G << 0.0f, -w.z,  w.y,
              w.z, 0.0f, -w.x,
             -w.y,  w.x, 0.0f;
        R = I + inv_theta * s * G
            + (1 - c) * inv_theta * inv_theta * G * G;
    }

    return R;
}


static mat4x4 solve_icp_p2p(JtJse3 J)
{
    Eigen::Matrix<float, 6, 6> A;
    Eigen::Matrix<float, 6, 1> b;
    Eigen::Matrix<float, 6, 1> x;

    b(0) = J.Jte[0];
    b(1) = J.Jte[1];
    b(2) = J.Jte[2];
    b(3) = J.Jte[3];
    b(4) = J.Jte[4];
    b(5) = J.Jte[5];

    A(0, 0) = J.JtJ[0];
    A(0, 1) = A(1, 0) = J.JtJ[1];
    A(0, 2) = A(2, 0) = J.JtJ[2];
    A(0, 3) = A(3, 0) = J.JtJ[3];
    A(0, 4) = A(4, 0) = J.JtJ[4];
    A(0, 5) = A(5, 0) = J.JtJ[5];
    A(1, 1) = J.JtJ[6];
    A(1, 2) = A(2, 1) = J.JtJ[7];
    A(1, 3) = A(3, 1) = J.JtJ[8];
    A(1, 4) = A(4, 1) = J.JtJ[9];
    A(1, 5) = A(5, 1) = J.JtJ[10];
    A(2, 2) = J.JtJ[11];
    A(2, 3) = A(3, 2) = J.JtJ[12];
    A(2, 4) = A(4, 2) = J.JtJ[13];
    A(2, 5) = A(5, 2) = J.JtJ[14];
    A(3, 3) = J.JtJ[15];
    A(3, 4) = A(4, 3) = J.JtJ[16];
    A(3, 5) = A(5, 3) = J.JtJ[17];
    A(4, 4) = J.JtJ[18];
    A(4, 5) = A(5, 4) = J.JtJ[19];
    A(5, 5) = J.JtJ[20];

    Eigen::JacobiSVD<Eigen::Matrix<float, 6, 6>>
        svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.solve(b);

    Eigen::Matrix3f R = rodrigues({x(3), x(4), x(5)});
    Eigen::Vector3f t = {x(0), x(1), x(2)};

    return mat4x4{R, t};
}


mat4x4 icp_p2p_se3(image<float3>* vm0, image<float4>* nm0, image<float3>* vm1, image<float4>* nm1,
                   intrinsics K, mat4x4 T, int num_iterations, float dist_threshold, float angle_threshold)
{
    static unsigned int reduce_size = 8;
    static unsigned int reduce_threads = 256;
    static image<JtJse3> JTJ, Axb;
    JTJ.resize(K.width, K.height, DEVICE_CUDA);
    Axb.resize(reduce_size, 1, DEVICE_CUDA_MAPPED);

    dim3 block_size(16, 16);
    dim3 grid_size;
    grid_size.x = divup(K.width, block_size.x);
    grid_size.y = divup(K.height, block_size.y);

    float last_error = FLT_MAX;
    for (int i = 0; i < num_iterations; ++i) {
        icp_p2p_se3_kernel<<<grid_size, block_size>>>(
            JTJ.gpu(), vm0->gpu(), nm0->gpu(), vm1->gpu(), nm1->gpu(),
            K, T, dist_threshold, angle_threshold);
        se3_reduce_kernel<<<reduce_size, reduce_threads, reduce_threads * sizeof(JtJse3)>>>(JTJ.gpu(), Axb.gpu());
        cudaDeviceSynchronize();

        for (int i = 1; i < reduce_size; ++i)
            Axb.data[0] += Axb.data[i];
        float error = Axb.data[0].error / Axb.data[0].weight;
        if (isnan(error) || error > last_error) break;

        mat4x4 delta = solve_icp_p2p(Axb.data[0]);
        T = delta * T;
        last_error = error;
    }
    return T;
}
