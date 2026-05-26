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

#include <modmesh/math/blas_compat.hpp>

#if defined(__APPLE__) && defined(__arm64__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64
#endif
#include <vecLib/cblas_new.h>

#elifdef MM_HAS_CBLAS
#include <cblas.h>
#endif

#include <complex>
#include <format>
#include <limits>
#include <stdexcept>

namespace modmesh
{

#if (defined(__APPLE__) && defined(__arm64__)) || defined(MM_HAS_CBLAS)
#if defined(__APPLE__) && defined(__arm64__)
using blas_int_type = __LAPACK_int;
#else
using blas_int_type = int;
#endif

static blas_int_type to_blas_int(size_t value, char const * name)
{
    if (value <= static_cast<size_t>(std::numeric_limits<blas_int_type>::max()))
    {
        return static_cast<blas_int_type>(value);
    }

    throw std::out_of_range(
        std::format("modmesh BLAS wrapper: {}={} exceeds BLAS integer range",
                    name,
                    value));
}

static CBLAS_TRANSPOSE to_cblas_transpose(bool transpose_matrix)
{
    return transpose_matrix ? CblasTrans : CblasNoTrans;
}

float dot_blas(size_t size, float const * lhs, float const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    return cblas_sdot(bsize, lhs, 1, rhs, 1);
}

double dot_blas(size_t size, double const * lhs, double const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    return cblas_ddot(bsize, lhs, 1, rhs, 1);
}

Complex<float> dot_blas(size_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    std::complex<float> result;
    cblas_cdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    1,
                    as_std_complex_pointer(rhs),
                    1,
                    &result);
    return result;
}

Complex<double> dot_blas(size_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    std::complex<double> result;
    cblas_zdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    1,
                    as_std_complex_pointer(rhs),
                    1,
                    &result);
    return result;
}

void gemv_blas(size_t m,
               size_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    cblas_sgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                1.0F,
                matrix,
                bn,
                vector,
                1,
                0.0F,
                result,
                1);
}

void gemv_blas(size_t m,
               size_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    cblas_dgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                1.0,
                matrix,
                bn,
                vector,
                1,
                0.0,
                result,
                1);
}

void gemv_blas(size_t m,
               size_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    cblas_cgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                &alpha,
                as_std_complex_pointer(matrix),
                bn,
                as_std_complex_pointer(vector),
                1,
                &beta,
                as_std_complex_pointer(result),
                1);
}

void gemv_blas(size_t m,
               size_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    cblas_zgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                &alpha,
                as_std_complex_pointer(matrix),
                bn,
                as_std_complex_pointer(vector),
                1,
                &beta,
                as_std_complex_pointer(result),
                1);
}

void gemm_blas(size_t m,
               size_t n,
               size_t k,
               float const * lhs,
               float const * rhs,
               float * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                1.0F,
                lhs,
                bk,
                rhs,
                bn,
                0.0F,
                result,
                bn);
}

void gemm_blas(size_t m,
               size_t n,
               size_t k,
               double const * lhs,
               double const * rhs,
               double * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                1.0,
                lhs,
                bk,
                rhs,
                bn,
                0.0,
                result,
                bn);
}

void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    cblas_cgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                &alpha,
                as_std_complex_pointer(lhs),
                bk,
                as_std_complex_pointer(rhs),
                bn,
                &beta,
                as_std_complex_pointer(result),
                bn);
}

void gemm_blas(size_t m,
               size_t n,
               size_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    cblas_zgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                &alpha,
                as_std_complex_pointer(lhs),
                bk,
                as_std_complex_pointer(rhs),
                bn,
                &beta,
                as_std_complex_pointer(result),
                bn);
}
#else
[[noreturn]] static void throw_blas_unavailable()
{
    throw std::runtime_error(
        "modmesh BLAS wrapper: CBLAS backend is unavailable");
}

float dot_blas(size_t, float const *, float const *)
{
    throw_blas_unavailable();
}

double dot_blas(size_t, double const *, double const *)
{
    throw_blas_unavailable();
}

Complex<float> dot_blas(size_t,
                        Complex<float> const *,
                        Complex<float> const *)
{
    throw_blas_unavailable();
}

Complex<double> dot_blas(size_t,
                         Complex<double> const *,
                         Complex<double> const *)
{
    throw_blas_unavailable();
}

void gemv_blas(size_t,
               size_t,
               float const *,
               float const *,
               float *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(size_t,
               size_t,
               double const *,
               double const *,
               double *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(size_t,
               size_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(size_t,
               size_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *,
               bool)
{
    throw_blas_unavailable();
}

void gemm_blas(size_t, size_t, size_t, float const *, float const *, float *)
{
    throw_blas_unavailable();
}

void gemm_blas(size_t,
               size_t,
               size_t,
               double const *,
               double const *,
               double *)
{
    throw_blas_unavailable();
}

void gemm_blas(size_t,
               size_t,
               size_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *)
{
    throw_blas_unavailable();
}

void gemm_blas(size_t,
               size_t,
               size_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *)
{
    throw_blas_unavailable();
}
#endif

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
