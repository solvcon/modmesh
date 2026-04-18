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

#include <modmesh/buffer/buffer.hpp>

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
    static array_type factorize(array_type const & a);
    // Solver: solve A x = b where A = L L^T (or L L^H for complex)
    static array_type solve(array_type const & a, array_type const & b);

private:

    // Forward substitution: solve L Y = B
    static array_type forward_substitution(array_type const & l, array_type const & b);
    // Backward substitution: solve L^T X = Y (or L^H X = Y for complex)
    static array_type backward_substitution(array_type const & l, array_type const & y);

}; /* end class Llt */

template <typename T>
auto Llt<T>::forward_substitution(array_type const & l, array_type const & b) -> array_type
{
    if (l.ndim() != 2 || l.shape(0) != l.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::forward_substitution: The first argument l must be a square 2D SimpleArray, but got shape (";
        for (ssize_t i = 0; i < l.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << l.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (b.ndim() != 2 || b.shape(0) != l.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::forward_substitution: The second argument b must be a 2D SimpleArray with first dimension matching l, but got shape (";
        for (ssize_t i = 0; i < b.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << b.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }

    const ssize_t m = static_cast<ssize_t>(l.shape(0));
    const ssize_t n = static_cast<ssize_t>(b.shape(1));
    small_vector<size_t> y_shape{static_cast<size_t>(m), static_cast<size_t>(n)};
    array_type y(y_shape);
    for (ssize_t k = 0; k < n; ++k)
    {
        for (ssize_t i = 0; i < m; ++i)
        {
            T sum = 0;
            for (ssize_t j = 0; j < i; ++j)
            {
                sum += l(i, j) * y(j, k);
            }
            y(i, k) = (b(i, k) - sum) / l(i, i);
        }
    }
    return y;
}

template <typename T>
auto Llt<T>::backward_substitution(array_type const & l, array_type const & y) -> array_type
{
    if (l.ndim() != 2 || l.shape(0) != l.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::backward_substitution: The first argument l must be a square 2D SimpleArray, but got shape (";
        for (ssize_t i = 0; i < l.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << l.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (y.ndim() != 2 || y.shape(0) != l.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::backward_substitution: The second argument y must be a 2D SimpleArray with first dimension matching l, but got shape (";
        for (ssize_t i = 0; i < y.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << y.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }

    const ssize_t m = static_cast<ssize_t>(l.shape(0));
    const ssize_t n = static_cast<ssize_t>(y.shape(1));
    small_vector<size_t> x_shape{static_cast<size_t>(m), static_cast<size_t>(n)};
    array_type x(x_shape);
    for (ssize_t k = 0; k < n; ++k)
    {
        for (ssize_t i = m - 1; i >= 0; --i)
        {
            T sum = 0;
            for (ssize_t j = i + 1; j < m; ++j)
            {
                sum += conj_mul(x(j, k), l(j, i));
            }
            x(i, k) = (y(i, k) - sum) / l(i, i);
        }
    }
    return x;
}

template <typename T>
auto Llt<T>::factorize(array_type const & a) -> array_type
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::factorize: The first argument a must be a square 2D SimpleArray, but got shape (";
        for (ssize_t i = 0; i < a.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << a.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }

    const ssize_t m = static_cast<ssize_t>(a.shape(0));
    small_vector<size_t> shape = {static_cast<size_t>(m), static_cast<size_t>(m)};
    array_type l(shape, value_type(0));
    const real_type eps = std::numeric_limits<real_type>::epsilon();
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j <= i; ++j)
        {
            value_type sum = 0;
            for (ssize_t k = 0; k < j; ++k)
            {
                sum += conj_mul(l(i, k), l(j, k));
            }
            if (i == j)
            {
                real_type dr = real(a(j, j) - sum);
                real_type tol = std::max<real_type>(1, abs(a(j, j))) * 100 * eps;
                if (dr <= tol)
                {
                    throw std::runtime_error("Llt::factorize: Cholesky failed: SimpleArray not (numerically) SPD.");
                }
                l(j, j) = std::sqrt(dr);
            }
            else
            {
                l(i, j) = (a(i, j) - sum) / l(j, j);
            }
        }
    }
    return l;
}

template <typename T>
auto Llt<T>::solve(array_type const & a, array_type const & b) -> array_type
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::solve: The first argument a must be a square 2D SimpleArray, but got shape (";
        for (ssize_t i = 0; i < a.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << a.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (a.shape(0) != b.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::solve: The first argument a and the second argument b dimension mismatch: a.shape[0]=" << a.shape(0) << ", b.shape[0]=" << b.shape(0);
        throw std::invalid_argument(oss.str());
    }
    if (b.ndim() != 1 && b.ndim() != 2)
    {
        std::ostringstream oss;
        oss << "Llt::solve: The second argument b must be 1D or 2D, but got " << b.ndim() << "D";
        throw std::invalid_argument(oss.str());
    }

    array_type l = factorize(a);

    bool was_1d = (b.ndim() == 1);
    if (was_1d)
    {
        array_type b_2d = b.reshape(small_vector<size_t>{b.shape(0), 1});
        array_type y = forward_substitution(l, b_2d);
        array_type x = backward_substitution(l, y);
        return x.reshape(small_vector<size_t>{x.shape(0)});
    }
    else
    {
        array_type y = forward_substitution(l, b);
        array_type x = backward_substitution(l, y);
        return x;
    }
}

} /* end namespace detail */

template <typename T>
SimpleArray<T> llt_factorization(SimpleArray<T> const & a)
{
    return detail::Llt<T>::factorize(a);
}

template <typename T>
SimpleArray<T> llt_solve(SimpleArray<T> const & a, SimpleArray<T> const & b)
{
    return detail::Llt<T>::solve(a, b);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: