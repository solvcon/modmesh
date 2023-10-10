/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/universe/bernstein.hpp>

namespace modmesh
{

namespace detail
{

template <typename T>
T calc_bernstein_polynomial_impl(T t, size_t i, size_t n)
{
    T ret = 1.0;
    {
        T const v = (i > 0) ? std::pow(t, i) : 1.0;
        ret *= v;
    }
    {
        T const v = (n > i) ? std::pow(1.0 - t, n - i) : 1.0;
        ret *= v;
    }
    for (size_t it = n; it > 1; --it)
    {
        ret *= it;
    }
    for (size_t it = i; it > 1; --it)
    {
        ret /= it;
    }
    for (size_t it = (n - i); it > 1; --it)
    {
        ret /= it;
    }
    return ret;
}

template <typename T>
T interpolate_bernstein_impl(T t, std::vector<T> const & values, size_t n)
{
    T ret = 0.0;
    for (size_t it = 0; it <= n; ++it)
    {
        T v = (it >= values.size()) ? 1.0 : values[it];
        v *= calc_bernstein_polynomial(t, it, n);
        ret += v;
    }
    return ret;
}

} /* end namespace detail */

double calc_bernstein_polynomial(double t, size_t i, size_t n)
{
    return detail::calc_bernstein_polynomial_impl<double>(t, i, n);
}

double interpolate_bernstein(double t, std::vector<double> const & values, size_t n)
{
    return detail::interpolate_bernstein_impl<double>(t, values, n);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
