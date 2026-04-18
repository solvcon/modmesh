/*
 * Copyright (c) 2025, Chun-Hsu Lai <as2266317@gmail.com>
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

#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <complex>

namespace modmesh
{

namespace detail
{

template <typename T>
struct ComplexImpl
{
    static_assert(std::is_floating_point_v<T>);

    T real_v;
    T imag_v;

    ComplexImpl() = default;

    ComplexImpl(T r, T i)
        : real_v(r)
        , imag_v(i)
    {
    }

    ComplexImpl(T t)
        : real_v(t)
        , imag_v(0.0)
    {
    }

    ComplexImpl(std::complex<T> const & c)
        : real_v(c.real())
        , imag_v(c.imag())
    {
    }

    ComplexImpl & operator+=(const ComplexImpl & other)
    {
        real_v += other.real_v;
        imag_v += other.imag_v;
        return *this;
    }

    ComplexImpl & operator+=(T other)
    {
        real_v += other;
        return *this;
    }

    ComplexImpl & operator-=(const ComplexImpl & other)
    {
        real_v -= other.real_v;
        imag_v -= other.imag_v;
        return *this;
    }

    ComplexImpl & operator-=(T other)
    {
        real_v -= other;
        return *this;
    }

    ComplexImpl & operator*=(const ComplexImpl<T> & rhs)
    {
        T real_v_copy = real_v;
        real_v = real_v_copy * rhs.real_v - imag_v * rhs.imag_v;
        imag_v = real_v_copy * rhs.imag_v + imag_v * rhs.real_v;
        return *this;
    }

    ComplexImpl & operator*=(const T & rhs)
    {
        real_v *= rhs;
        imag_v *= rhs;
        return *this;
    }

    ComplexImpl & operator/=(const ComplexImpl<T> & rhs)
    {
        T denominator = rhs.norm();
        T real_v_copy = real_v;

        if (denominator == 0.0)
        {
            throw std::runtime_error("Division by zero in complex number");
        }

        real_v = (real_v * rhs.real_v + imag_v * rhs.imag_v) / denominator;
        imag_v = (imag_v * rhs.real_v - real_v_copy * rhs.imag_v) / denominator;
        return *this;
    }

    ComplexImpl & operator/=(const T & rhs)
    {
        if (rhs == 0.0)
        {
            throw std::runtime_error("Division by zero in complex number");
        }
        real_v /= rhs;
        imag_v /= rhs;
        return *this;
    }

    std::complex<T> to_std_complex() const { return std::complex<T>(real_v, imag_v); }
    T real() const { return real_v; }
    T imag() const { return imag_v; }
    T norm() const { return real_v * real_v + imag_v * imag_v; }
    ComplexImpl<T> conj() const { return {real_v, -imag_v}; }
}; /* end struct ComplexImpl */

/**
 * These comparison operators use lexicographic ordering and would be used in
 * SimpleArray::min() and SimpleArray::max(). The use of lexicographic ordering
 * is to match the numpy behaviors documented in
 * https://numpy.org/devdocs/reference/generated/numpy.sort.html . The
 * following discussions include more details:
 * more details:
 * 1. https://github.com/numpy/numpy/issues/12943
 * 2. https://stackoverflow.com/questions/52481376
 */
template <typename T>
bool operator<(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    if (lhs.real_v == rhs.real_v)
    {
        return lhs.imag_v <= rhs.imag_v;
    }
    return lhs.real_v <= rhs.real_v;
}

template <typename T>
bool operator>(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    if (lhs.real_v == rhs.real_v)
    {
        return lhs.imag_v > rhs.imag_v;
    }
    return lhs.real_v > rhs.real_v;
}

template <typename T>
bool operator==(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    return lhs.real_v == rhs.real_v && lhs.imag_v == rhs.imag_v;
}

template <typename T>
bool operator!=(const ComplexImpl<T> & lhs, const ComplexImpl<T> & rhs)
{
    return lhs.real_v != rhs.real_v || lhs.imag_v != rhs.imag_v;
}

template <typename T>
ComplexImpl<T> operator+(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs += rhs;
}

template <typename T>
ComplexImpl<T> operator+(ComplexImpl<T> lhs, T rhs)
{
    return lhs += rhs;
}

template <typename T>
ComplexImpl<T> operator+(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} += rhs;
}

template <typename T>
ComplexImpl<T> operator-(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs -= rhs;
}

template <typename T>
ComplexImpl<T> operator-(ComplexImpl<T> lhs, T rhs)
{
    return lhs -= rhs;
}

template <typename T>
ComplexImpl<T> operator-(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} -= rhs;
}

template <typename T>
ComplexImpl<T> operator*(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs *= rhs;
}

template <typename T>
ComplexImpl<T> operator*(ComplexImpl<T> lhs, T rhs)
{
    return lhs *= rhs;
}

template <typename T>
ComplexImpl<T> operator*(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} *= rhs;
}

template <typename T>
ComplexImpl<T> operator/(ComplexImpl<T> lhs, const ComplexImpl<T> & rhs)
{
    return lhs /= rhs;
}

template <typename T>
ComplexImpl<T> operator/(ComplexImpl<T> lhs, const T & rhs)
{
    return lhs /= rhs;
}

template <typename T>
ComplexImpl<T> operator/(T lhs, const ComplexImpl<T> & rhs)
{
    return ComplexImpl<T>{lhs, 0.0} /= rhs;
}

} /* end namespace detail */

template <typename T>
using Complex = detail::ComplexImpl<T>;

// clang-format off
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<Complex<T>> : std::true_type {};
// clang-format on

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
