#pragma once

#if defined(__aarch64__)

#include <cstddef>
#include <arm_neon.h>

namespace modmesh
{

namespace simd
{

namespace neon
{

namespace type
{

namespace detail
{

template <typename T>
struct vector
{
    static constexpr size_t N_lane = 0;
};

template <>
struct vector<uint8_t>
{
    using type = uint8x16_t;
    static constexpr size_t N_lane = 16;
};

template <>
struct vector<uint16_t>
{
    using type = uint16x8_t;
    static constexpr size_t N_lane = 8;
};

template <>
struct vector<uint32_t>
{
    using type = uint32x4_t;
    static constexpr size_t N_lane = 4;
};

template <>
struct vector<uint64_t>
{
    using type = uint64x2_t;
    static constexpr size_t N_lane = 2;
};

template <>
struct vector<int8_t>
{
    using type = int8x16_t;
    static constexpr size_t N_lane = 16;
};

template <>
struct vector<int16_t>
{
    using type = int16x8_t;
    static constexpr size_t N_lane = 8;
};

template <>
struct vector<int32_t>
{
    using type = int32x4_t;
    static constexpr size_t N_lane = 4;
};

template <>
struct vector<int64_t>
{
    using type = int64x2_t;
    static constexpr size_t N_lane = 2;
};

} /* namespace detail */

template <typename T>
using vector_t = typename detail::vector<T>::type;

template <typename T>
inline constexpr size_t vector_lane = detail::vector<T>::N_lane;

template <typename T>
inline constexpr size_t has_vectype = detail::vector<T>::N_lane > 0;

} /* namespace type */

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */

#endif /* defined(__aarch64__) */
