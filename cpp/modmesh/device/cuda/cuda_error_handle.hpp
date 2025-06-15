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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#define CUDA_SAFE_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUFFT_SAFE_CALL(err) __cufftSafeCall(err, __FILE__, __LINE__)
#define CUDA_GET_LAST_ERROR() __cudaCheckError(__FILE__, __LINE__)

void inline __cudaSafeCall(cudaError_t err, const char * file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }
}

void inline __cudaCheckError(const char * file, const int line)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }
}

const inline char * __cufftResultToString(cufftResult err)
{
    switch (err)
    {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS.";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN.";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED.";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE.";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE.";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR.";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED.";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED.";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE.";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA.";
    default: return "CUFFT Unknown error code.";
    }
}

void inline __cufftSafeCall(cufftResult err, const char * file, const int line)
{
    if (CUFFT_SUCCESS != err)
    {
        printf("CUFFT error %d: %s\n%s(%d)\n", (int)err, __cufftResultToString(err), file, line);
    }
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
