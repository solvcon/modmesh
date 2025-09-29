#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/math/Complex.hpp>

#include <type_traits>
#include <cmath>

namespace modmesh
{

namespace detail
{

template <typename T>
inline constexpr T pow(T /*base*/, std::integral_constant<size_t, 0> /*unused*/) { return 1; }

template <typename T>
inline constexpr T pow(T base, std::integral_constant<size_t, 1> /*unused*/) { return base; }

template <typename T, size_t N>
inline constexpr T pow(T base, std::integral_constant<size_t, N> /*unused*/)
{
    return pow(base, std::integral_constant<size_t, N - 1>()) * base;
}

template <typename T>
constexpr T pi_v = std::enable_if_t<std::is_floating_point_v<T>, T>(3.141592653589793238462643383279502884L);

} /* end namespace detail */

template <size_t N, typename T>
inline constexpr T pow(T base)
{
    return detail::pow(base, std::integral_constant<size_t, N>());
}

template <typename T>
constexpr T pi = detail::pi_v<T>;

template <typename T>
inline constexpr T conj_mul(T const & a, T const & b)
{
    if constexpr (is_complex_v<T>)
    {
        return a * b.conj();
    }
    else
    {
        return a * b;
    }
}

template <typename T>
inline auto real(T const & val)
{
    if constexpr (is_complex_v<T>)
    {
        return val.real();
    }
    else
    {
        return val;
    }
}

template <typename T>
inline auto abs(T const & val)
{
    if constexpr (is_complex_v<T>)
    {
        return std::sqrt(val.norm());
    }
    else
    {
        return std::abs(val);
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
