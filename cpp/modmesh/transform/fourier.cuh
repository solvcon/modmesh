#pragma once

/*
 * Copyright (c) 2025, Alex Chiang <jyxemperor@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/modmesh.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/device/cuda/cuda_error_handle.hpp>

#define FFT_CUDA_IMPL(CUFFT_DATA_TYPE, CUFFT_EXEC_TYPE)                                                    \
{                                                                                                          \
    cufftHandle plan;                                                                                      \
    CUFFT_DATA_TYPE * host_in = nullptr;                                                                   \
    CUFFT_DATA_TYPE * host_out = nullptr;                                                                  \
    CUFFT_DATA_TYPE * device_in = nullptr;                                                                 \
    CUFFT_DATA_TYPE * device_out = nullptr;                                                                \
    host_in = (CUFFT_DATA_TYPE*)malloc(sizeof(CUFFT_DATA_TYPE) * N);                                       \
    host_out = (CUFFT_DATA_TYPE*)malloc(sizeof(CUFFT_DATA_TYPE) * N);                                      \
    for (size_t i = 0; i < N; ++i)                                                                         \
    {                                                                                                      \
        host_in[i].x = in[i].real();                                                                       \
        host_in[i].y = in[i].imag();                                                                       \
    }                                                                                                      \
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_in, sizeof(CUFFT_DATA_TYPE) * N));                           \
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_out, sizeof(CUFFT_DATA_TYPE) * N));                          \
    CUDA_SAFE_CALL(cudaMemcpy(device_in, host_in, sizeof(CUFFT_DATA_TYPE) * N, cudaMemcpyHostToDevice));   \
    CUFFT_SAFE_CALL(cufftPlan1d(&plan, N, CUFFT_##CUFFT_EXEC_TYPE, 1));                                    \
    CUFFT_SAFE_CALL(cufftExec##CUFFT_EXEC_TYPE(plan, device_in, device_out, CUFFT_FORWARD));               \
    CUDA_SAFE_CALL(cudaMemcpy(host_out, device_out, sizeof(CUFFT_DATA_TYPE) * N, cudaMemcpyDeviceToHost)); \
    for (size_t i = 0; i < N; ++i)                                                                         \
    {                                                                                                      \
        out[i] = T1<T2>{ host_out[i].x, host_out[i].y };                                                   \
    }                                                                                                      \
    CUFFT_SAFE_CALL(cufftDestroy(plan));                                                                   \
    CUDA_SAFE_CALL(cudaFree(device_in));                                                                   \
    CUDA_SAFE_CALL(cudaFree(device_out));                                                                  \
    free(host_in);                                                                                         \
    free(host_out);                                                                                        \
}

namespace modmesh
{

template <template <typename> class T1, typename T2>
void fft_cuda(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
{
    size_t N = in.size();
    if constexpr (std::is_same_v<T2, float>)
    {
        FFT_CUDA_IMPL(cufftComplex, C2C)
    }
    else if constexpr (std::is_same_v<T2, double>)
    {
        FFT_CUDA_IMPL(cufftDoubleComplex, Z2Z)
    }
}

} /* namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
