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
