#pragma once

/*
 * Copyright (c) 2025, Chun-Shih Chang <austin20463@gmail.com>
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

#include <modmesh/buffer/SimpleArray.hpp>

namespace modmesh
{

namespace detail
{

template <typename T>
class Llt
{

private:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<value_type>;

public:

    // Factorization algorithm
    // Reference: https://www.geeksforgeeks.org/dsa/cholesky-decomposition-matrix-decomposition/
    static array_type factorize(array_type const & A);
    // Solver: solve A x = b where A = L L^T (or L L^H for complex)
    static array_type solve(array_type const & A, array_type const & b);

private:

    // Forward substitution: solve L y = b
    static array_type forward_substitution(array_type const & L, array_type const & b);
    // Backward substitution: solve L^T x = y (or L^H x = y for complex)
    static array_type backward_substitution(array_type const & L, array_type const & y);

}; /* end class Llt */

template <typename T>
auto Llt<T>::forward_substitution(array_type const & L, array_type const & b) -> array_type
{
    const size_t n = L.shape(0);
    array_type y(n);
    for (size_t i = 0; i < n; ++i)
    {
        T sum = 0;
        for (size_t j = 0; j < i; ++j)
        {
            sum += L(i, j) * y(j);
        }
        y(i) = (b(i) - sum) / L(i, i);
    }
    return y;
}

template <typename T>
auto Llt<T>::backward_substitution(array_type const & L, array_type const & y) -> array_type
{
    const size_t n = L.shape(0);
    array_type x(n);
    for (int i = n - 1; i >= 0; --i)
    {
        T sum = 0;
        for (size_t j = i + 1; j < n; ++j)
        {
            sum += conj_mul(x(j), L(j, i));
        }
        x(i) = (y(i) - sum) / L(i, i);
    }
    return x;
}

template <typename T>
auto Llt<T>::factorize(array_type const & A) -> array_type
{
    if (A.ndim() != 2 || A.shape(0) != A.shape(1))
    {
        throw std::invalid_argument("Llt factorization requires a square 2D matrix");
    }

    const size_t n = A.shape(0);
    small_vector<size_t> shape = {n, n};
    array_type L(shape, value_type(0));
    const real_type eps = std::numeric_limits<real_type>::epsilon();
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            value_type sum = 0;
            for (size_t k = 0; k < j; ++k)
            {
                sum += conj_mul(L(i, k), L(j, k));
            }
            if (i == j)
            {
                real_type dr = real(A(j, j) - sum);
                real_type tol = std::max<real_type>(1, abs(A(j, j))) * 100 * eps;
                if (dr <= tol)
                {
                    throw std::runtime_error("Cholesky failed: matrix not (numerically) SPD.");
                }
                L(j, j) = std::sqrt(dr);
            }
            else
            {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
    return L;
}

template <typename T>
auto Llt<T>::solve(array_type const & A, array_type const & b) -> array_type
{
    if (A.ndim() != 2 || A.shape(0) != A.shape(1))
    {
        throw std::invalid_argument("Matrix A must be square 2D");
    }
    if (b.ndim() != 1)
    {
        throw std::invalid_argument("Vector b must be 1D");
    }
    if (A.shape(0) != b.shape(0))
    {
        throw std::invalid_argument("Matrix A and vector b dimension mismatch");
    }
    array_type L = factorize(A);
    array_type y = forward_substitution(L, b);
    array_type x = backward_substitution(L, y);
    return x;
}

} /* end namespace detail */

template <typename T>
SimpleArray<T> llt_factorization(SimpleArray<T> const & A)
{
    return detail::Llt<T>::factorize(A);
}

template <typename T>
SimpleArray<T> llt_solve(SimpleArray<T> const & A, SimpleArray<T> const & b)
{
    return detail::Llt<T>::solve(A, b);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: