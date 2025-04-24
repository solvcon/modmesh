#pragma once

#if defined(__aarch64__)

#include <type_traits>
#include <arm_neon.h>
#include <modmesh/simd/neon/neon_type.hpp>

namespace modmesh
{

namespace simd
{

namespace neon
{

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vdupq(base_type val);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vld1q(base_type * ptr);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vld1q(base_type const * ptr);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
void vst1q(base_type * ptr, type::vector_t<base_type> vec);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vcgeq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vcltq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

#define utype_t(N) uint##N##_t
#define stype_t(N) int##N##_t
#define DECL_MM_IMPL_VGETQ(N)                                                                                          \
    template <typename base_type, size_t n, typename std::enable_if_t<std::is_same_v<utype_t(N), base_type>, int> = 0> \
    base_type vgetq(type::vector_t<base_type> vec)                                                                     \
    {                                                                                                                  \
        return vgetq_lane_u##N(vec, n);                                                                                \
    }                                                                                                                  \
    template <typename base_type, size_t n, typename std::enable_if_t<std::is_same_v<stype_t(N), base_type>, int> = 0> \
    base_type vgetq(type::vector_t<base_type> vec)                                                                     \
    {                                                                                                                  \
        return vgetq_lane_s##N(vec, n);                                                                                \
    }

DECL_MM_IMPL_VGETQ(8)
DECL_MM_IMPL_VGETQ(16)
DECL_MM_IMPL_VGETQ(32)
DECL_MM_IMPL_VGETQ(64)

#undef DECL_MM_IMPL_VGETQ
#undef stype_t
#undef utype_t

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */

#endif /* defined(__aarch64__) */
