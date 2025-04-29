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

#include <modmesh/simd/simd_generic.hpp>
#include <modmesh/simd/neon/neon_type.hpp>
#include <modmesh/simd/neon/neon_alias.hpp>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif /* defined(__aarch64__) */

namespace modmesh
{

namespace simd
{

namespace neon
{

#if defined(__aarch64__)
template <typename T, typename std::enable_if_t<!type::has_vectype<T>> * = nullptr>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    return generic::check_between<T>(start, end, min_val, max_val);
}

template <typename T, typename std::enable_if_t<type::has_vectype<T>> * = nullptr>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    using vec_t = type::vector_t<T>;
    using cmpvec_t = type::vector_t<uint64_t>;
    constexpr size_t N_lane = type::vector_lane<T>;

    vec_t max_vec = vdupq<T>(max_val);
    vec_t min_vec = vdupq<T>(min_val);
    vec_t data_vec = {};
    cmpvec_t cmp_vec = {};
    T const * ret = NULL;

    T const * ptr = start;
    for (; ptr <= end - N_lane; ptr += N_lane)
    {
        data_vec = vld1q<T>(ptr);
        cmp_vec = (cmpvec_t)vcgeq<T>(data_vec, max_vec);
        if (vgetq<uint64_t, 0>(cmp_vec) ||
            vgetq<uint64_t, 1>(cmp_vec))
        {
            goto OUT_OF_RANGE;
        }

        cmp_vec = (cmpvec_t)vcltq<T>(data_vec, min_vec);
        if (vgetq<uint64_t, 0>(cmp_vec) ||
            vgetq<uint64_t, 1>(cmp_vec))
        {
            goto OUT_OF_RANGE;
        }
    }

    if (ptr != end)
    {
        ret = generic::check_between<T>(ptr, end, min_val, max_val);
    }

    return ret;

OUT_OF_RANGE:
    T cmp_val[N_lane] = {};
    T * cmp = cmp_val;
    vst1q<T>(cmp_val, cmp_vec);

    for (size_t i = 0; i < N_lane; ++i, ++cmp)
    {
        if (*cmp)
        {
            return ptr + i;
        }
    }
    return ptr;
}
#else
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    return generic::check_between<T>(start, end, min_val, max_val);
}
#endif /* defined(__aarch64__) */

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */
