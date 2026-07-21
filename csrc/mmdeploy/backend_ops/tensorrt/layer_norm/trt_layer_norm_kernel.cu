// Source: https://github.com/NVIDIA/TensorRT/tree/8.5.3/plugin/layerNormPlugin
// Modified by romiptr
#include <assert.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "cub/cub.cuh"
#include "common_cuda_helper.hpp"
#include "trt_layer_norm_kernel.hpp"
#include "trt_plugin_helper.hpp"

// Copied from common/common.cuh (copy<>)
// https://github.com/NVIDIA/TensorRT/blob/8.5.3/plugin/common/common.cuh#L453
template <int Bytes>
struct BytesToType;
template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint2;
};
template <>
struct BytesToType<16>
{
    using type = uint4;
};

template <int Bytes>
__device__ inline void copy(void const* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
    T const* in = static_cast<T const*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

// explicit declaration in float type because in the original source code is only using float 
using kvp = cub::KeyValuePair<float, float>;

struct mySum
{
    __host__ __device__ __forceinline__ kvp operator()(const kvp& a, const kvp& b) const
    {
        return kvp(a.key + b.key, a.value + b.value);
    }
};

template <typename scalar_t, int TPB>
__global__ void LayerNormSmallKernel(
    const int nHiddenDimension, const float denominator, const scalar_t* input, 
    const scalar_t* gamma, const scalar_t* beta, scalar_t* output, const float epsilon)
{
    const int index = blockIdx.x * nHiddenDimension + threadIdx.x;
    float val = 0;
    kvp threadData(0, 0);

    if (threadIdx.x < nHiddenDimension)
    {
        val = static_cast<float>(input[index]) * denominator;
        float tmp0 = val * denominator;
        float tmp1 = val * tmp0;
        threadData = mySum()(threadData, kvp(tmp0, tmp1));
    }

    using WarpReduce = cub::WarpReduce<kvp, TPB>;
    __shared__ typename WarpReduce::TempStorage temp;
    __shared__ float mu, rsigma;

    auto const sumKV = WarpReduce(temp).Reduce(threadData, mySum());
    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + static_cast<float>(epsilon));
    }
    __syncthreads();

    if (threadIdx.x < nHiddenDimension)
    {
        const float g = gamma[threadIdx.x], b = beta[threadIdx.x];
        output[index] = (val - mu) * rsigma * g + b;
    }
}

template <typename scalar_t, int TPB, int VPT>
__global__ void LayerNormMediumKernel(
    const int nHiddenDimension, const float denominator, const scalar_t* input, 
    const scalar_t* gamma, const scalar_t* beta, scalar_t* output, const float epsilon)
{
    const int index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
    scalar_t localX[VPT], localGamma[VPT], localBeta[VPT];
    kvp threadData(0, 0);

    copy<sizeof(scalar_t) * VPT>(&input[index], localX);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = static_cast<float>(localX[it]) * denominator;
        threadData = mySum()(threadData, kvp(tmp, tmp * static_cast<float>(localX[it])));
    }

    copy<sizeof(scalar_t) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(scalar_t) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

    using BlockReduce = cub::BlockReduce<kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu, rsigma;

    auto const sumKV = BlockReduce(temp_storage).Reduce(threadData, mySum());
    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + static_cast<float>(epsilon));
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = static_cast<float>(localGamma[it]) * (static_cast<float>(localX[it]) - mu) * rsigma + static_cast<float>(localBeta[it]);
    }

    copy<sizeof(scalar_t) * VPT>(localX, &output[index]);
}

template <typename scalar_t, int TPB>
__global__ void LayerNormLargeKernel(
    const int nHiddenDimension, const float denominator, const scalar_t* input, 
    const scalar_t* gamma, const scalar_t* beta, scalar_t* output, const float epsilon)
{
    const int offset = blockIdx.x * nHiddenDimension;
    kvp threadData(0, 0);

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int index = offset + i;
        float val = input[index];
        const float tmp = val * denominator;
        threadData = mySum()(threadData, kvp(tmp, tmp * val));
        output[index] = val;
    }

    using BlockReduce = cub::BlockReduce<kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float mu, rsigma;

    const auto sumKV = BlockReduce(temp).Reduce(threadData, mySum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + static_cast<float>(epsilon));
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int index = offset + i;
        output[index] = (static_cast<float>(output[index]) - mu) * rsigma * static_cast<float>(gamma[i]) + static_cast<float>(beta[i]);
    }
}

template <typename scalar_t>
void TRTLayerNormalizationKernelLauncher(
    const int gridSize, const int nHiddenDimension, const scalar_t* input, const scalar_t* gamma,
    const scalar_t* beta, scalar_t* output, const float epsilon, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(scalar_t);
    const float denominator = 1.0f / static_cast<float>(nHiddenDimension);

    if (nHiddenDimension <= 32)
    {
        constexpr int TPB = 32;
        (LayerNormSmallKernel<scalar_t, TPB>) <<<gridSize, TPB, 0, stream>>>(
            nHiddenDimension, denominator, input, gamma, beta, output, epsilon);
    }
    else if (nHiddenDimension == 320)
    {
        constexpr int TPB = 320 / VPT;
        (LayerNormMediumKernel<scalar_t, TPB, VPT>) <<<gridSize, TPB, 0, stream>>>(
            nHiddenDimension, denominator, input, gamma, beta, output, epsilon);
    }
    else if (nHiddenDimension == 640)
    {
        constexpr int TPB = 640 / VPT;
        (LayerNormMediumKernel<scalar_t, TPB, VPT>) <<<gridSize, TPB, 0, stream>>>(
            nHiddenDimension, denominator, input, gamma, beta, output, epsilon);
    }
    else if (nHiddenDimension == 768)
    {
        constexpr int TPB = 768 / VPT;
        (LayerNormMediumKernel<scalar_t, TPB, VPT>) <<<gridSize, TPB, 0, stream>>>(
            nHiddenDimension, denominator, input, gamma, beta, output, epsilon);
    }
    else
    {
        constexpr int TPB = 256;
        (LayerNormLargeKernel<scalar_t, TPB>) <<<gridSize, TPB, 0, stream>>>(
            nHiddenDimension, denominator, input, gamma, beta, output, epsilon);
    }
    cudaCheckError();
}

template void TRTLayerNormalizationKernelLauncher<float>(
    const int gridSize, const int nHiddenDimension, const float* input, const float* gamma, 
    const float* beta, float* output, const float epsilon, cudaStream_t stream);
template void TRTLayerNormalizationKernelLauncher<__half>(
    const int gridSize, const int nHiddenDimension, const __half* input, const __half* gamma, 
    const __half* beta, __half* output, const float epsilon, cudaStream_t stream);
