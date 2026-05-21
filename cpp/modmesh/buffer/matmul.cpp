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

#include <modmesh/buffer/matmul.hpp>

#if defined(__APPLE__) && defined(__arm64__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64
#endif
#include <vecLib/cblas_new.h>

#include <format>
#include <limits>
#endif

#include <stdexcept>

namespace modmesh
{

namespace detail
{

#if defined(__APPLE__) && defined(__arm64__)
struct BlasDims
{
    BlasDims(size_t m_in, size_t n_in, size_t k_in)
        : m(to_lapack_int(m_in, "m"))
        , n(to_lapack_int(n_in, "n"))
        , k(to_lapack_int(k_in, "k"))
    {
    }

    __LAPACK_int m;
    __LAPACK_int n;
    __LAPACK_int k;

private:

    static __LAPACK_int to_lapack_int(size_t value, char const * name)
    {
        if (value <= static_cast<size_t>(std::numeric_limits<__LAPACK_int>::max()))
        {
            return static_cast<__LAPACK_int>(value);
        }

        throw std::out_of_range(
            std::format("SimpleArray::matmul_veclib(): {}={} exceeds LAPACK integer range",
                        name,
                        value));
    }
};

void matmul_veclib_backend(size_t m,
                           size_t n,
                           size_t k,
                           float const * lhs,
                           float const * rhs,
                           float * result)
{
    BlasDims const dims(m, n, k);
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dims.m,
                dims.n,
                dims.k,
                1.0F,
                lhs,
                dims.k,
                rhs,
                dims.n,
                0.0F,
                result,
                dims.n);
}

void matmul_veclib_backend(size_t m,
                           size_t n,
                           size_t k,
                           double const * lhs,
                           double const * rhs,
                           double * result)
{
    BlasDims const dims(m, n, k);
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dims.m,
                dims.n,
                dims.k,
                1.0,
                lhs,
                dims.k,
                rhs,
                dims.n,
                0.0,
                result,
                dims.n);
}

void matmul_veclib_backend(size_t m,
                           size_t n,
                           size_t k,
                           Complex<float> const * lhs,
                           Complex<float> const * rhs,
                           Complex<float> * result)
{
    BlasDims const dims(m, n, k);
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    cblas_cgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dims.m,
                dims.n,
                dims.k,
                &alpha,
                as_std_complex_pointer(lhs),
                dims.k,
                as_std_complex_pointer(rhs),
                dims.n,
                &beta,
                as_std_complex_pointer(result),
                dims.n);
}

void matmul_veclib_backend(size_t m,
                           size_t n,
                           size_t k,
                           Complex<double> const * lhs,
                           Complex<double> const * rhs,
                           Complex<double> * result)
{
    BlasDims const dims(m, n, k);
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    cblas_zgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dims.m,
                dims.n,
                dims.k,
                &alpha,
                as_std_complex_pointer(lhs),
                dims.k,
                as_std_complex_pointer(rhs),
                dims.n,
                &beta,
                as_std_complex_pointer(result),
                dims.n);
}
#else
void matmul_veclib_backend(size_t, size_t, size_t, float const *, float const *, float *)
{
    throw_matmul_veclib_unavailable();
}

void matmul_veclib_backend(size_t, size_t, size_t, double const *, double const *, double *)
{
    throw_matmul_veclib_unavailable();
}

void matmul_veclib_backend(size_t, size_t, size_t, Complex<float> const *, Complex<float> const *, Complex<float> *)
{
    throw_matmul_veclib_unavailable();
}

void matmul_veclib_backend(size_t, size_t, size_t, Complex<double> const *, Complex<double> const *, Complex<double> *)
{
    throw_matmul_veclib_unavailable();
}
#endif

} /* end namespace detail */

} /* end namespace modmesh */
