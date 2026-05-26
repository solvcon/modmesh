#pragma once

/*
 * Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
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

#include <modmesh/math/Complex.hpp>

#include <cstddef>

namespace modmesh
{

float dot_blas(size_t size, float const * lhs, float const * rhs);
double dot_blas(size_t size, double const * lhs, double const * rhs);
Complex<float> dot_blas(size_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs);
Complex<double> dot_blas(size_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs);
void gemv_blas(size_t m,
               size_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix);
void gemv_blas(size_t m,
               size_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               float const * lhs,
               float const * rhs,
               float * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               double const * lhs,
               double const * rhs,
               double * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result);
void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result);

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
