#pragma once

/*
 * Copyright (c) 2025, Kuan-Hsien Lee <khlee870529@gmail.com>
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
// clang-format off
#define DECL_MM_IMPL_VGETQ(N)                                                           \
    template <typename base_type, size_t n,                                             \
        typename std::enable_if_t<                                                      \
            n < type::vector_lane<utype_t(N)> && std::is_same_v<utype_t(N), base_type>  \
        > * = nullptr>                                                                  \
    base_type vgetq(type::vector_t<base_type> vec)                                      \
    {                                                                                   \
        return vgetq_lane_u##N(vec, n);                                                 \
    }                                                                                   \
    template <typename base_type, size_t n,                                             \
        typename std::enable_if_t<                                                      \
            n < type::vector_lane<stype_t(N)> && std::is_same_v<stype_t(N), base_type>  \
        > * = nullptr>                                                                  \
    base_type vgetq(type::vector_t<base_type> vec)                                      \
    {                                                                                   \
        return vgetq_lane_s##N(vec, n);                                                 \
    }
// clang-format on

DECL_MM_IMPL_VGETQ(8)
DECL_MM_IMPL_VGETQ(16)
DECL_MM_IMPL_VGETQ(32)
DECL_MM_IMPL_VGETQ(64)

#undef DECL_MM_IMPL_VGETQ
#undef stype_t
#undef utype_t

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vaddq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

template <typename base_type, typename = std::enable_if_t<type::has_vectype<base_type>>>
type::vector_t<base_type> vsubq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

template <typename base_type, typename = std::enable_if_t<2 < type::vector_lane<base_type> || std::is_floating_point_v<base_type>>>
type::vector_t<base_type> vmulq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

template <typename base_type, typename = std::enable_if_t<std::is_floating_point_v<base_type>>>
type::vector_t<base_type> vdivq(type::vector_t<base_type> vec_a, type::vector_t<base_type> vec_b);

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */

#endif /* defined(__aarch64__) */
