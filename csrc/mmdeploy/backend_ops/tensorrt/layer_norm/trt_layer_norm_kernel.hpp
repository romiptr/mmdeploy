// Source: https://github.com/NVIDIA/TensorRT/tree/8.5.3/plugin/layerNormPlugin
// Modified by romiptr
#ifndef TRT_LAYERNORM_KERNEL_HPP
#define TRT_LAYERNORM_KERNEL_HPP
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
void TRTLayerNormalizationKernelLauncher(
    const int gridSize, const int nHiddenDimension, const scalar_t* input, const scalar_t* gamma, 
    const scalar_t* beta, scalar_t* output, const float epsilon, cudaStream_t stream);

#endif // TRT_LAYERNORM_KERNEL_HPP