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

namespace detail
{

#ifndef NDEBUG
template <typename T>
bool is_aligned(T const * pointer, size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
}

template <typename T>
void check_alignment(T const * pointer, size_t required_alignment, const char * name)
{
    if (!is_aligned(pointer, required_alignment))
    {
        std::fprintf(stderr,
                     "Warning: %s pointer %p is not aligned to %zu bytes. "
                     "SIMD performance may be degraded.\n",
                     name,
                     static_cast<const void *>(pointer),
                     required_alignment);
    }
}
#endif // NDEBUG

// Get the recommended memory alignment for SIMD operations based on the detected SIMD instruction set.
inline constexpr size_t get_recommended_alignment()
{
#if defined(__aarch64__) || defined(__arm__)
    return 16;
// TODO: The non-NEON conditional should be factored out elsewhere in the future.
#elif defined(__AVX512F__)
    return 64;
#elif defined(__AVX__) || defined(__AVX2__)
    return 32;
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
    return 16;
#else
    return 0;
#endif
}

} /* end namespace detail */

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

#ifndef NDEBUG
    constexpr size_t alignment = detail::get_recommended_alignment();
    detail::check_alignment(start, alignment, "check_between start");
#endif

    vec_t max_vec = vdupq(max_val);
    vec_t min_vec = vdupq(min_val);
    vec_t data_vec = {};
    cmpvec_t cmp_vec = {};
    T const * ret = NULL;

    T const * ptr = start;

    // Check if array is large enough for SIMD processing
    if (end - start < N_lane)
    {
        return generic::check_between<T>(start, end, min_val, max_val);
    }

    for (; ptr <= end - N_lane; ptr += N_lane)
    {
        data_vec = vld1q(ptr);
        cmp_vec = (cmpvec_t)vcgeq(data_vec, max_vec);
        if (vgetq<0>(cmp_vec) ||
            vgetq<1>(cmp_vec))
        {
            goto OUT_OF_RANGE;
        }

        cmp_vec = (cmpvec_t)vcltq(data_vec, min_vec);
        if (vgetq<0>(cmp_vec) ||
            vgetq<1>(cmp_vec))
        {
            goto OUT_OF_RANGE;
        }
    }

    if (ptr != end)
    {
        ret = check_between<T>(ptr, end, min_val, max_val);
    }

    return ret;

OUT_OF_RANGE:
    T cmp_val[N_lane] = {};
    T * cmp = cmp_val;
    vst1q(cmp_val, cmp_vec);

    for (size_t i = 0; i < N_lane; ++i, ++cmp)
    {
        if (*cmp)
        {
            return ptr + i;
        }
    }
    return ptr;
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    if constexpr (!(type::has_vectype<T>))
    {
        return generic::add<T>(dest, dest_end, src1, src2);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
        constexpr size_t alignment = detail::get_recommended_alignment();
        detail::check_alignment(dest, alignment, "add dest");
        detail::check_alignment(src1, alignment, "add src1");
        detail::check_alignment(src2, alignment, "add src2");
#endif

        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q(src1);
            src2_vec = vld1q(src2);
            res_vec = vaddq(src1_vec, src2_vec);
            vst1q(ptr, res_vec);
        }
        if (ptr != dest_end)
        {
            generic::add<T>(ptr, dest_end, src1, src2);
        }
    }
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    if constexpr (!(type::has_vectype<T>))
    {
        return generic::sub<T>(dest, dest_end, src1, src2);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
        constexpr size_t alignment = detail::get_recommended_alignment();
        detail::check_alignment(dest, alignment, "sub dest");
        detail::check_alignment(src1, alignment, "sub src1");
        detail::check_alignment(src2, alignment, "sub src2");
#endif

        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q(src1);
            src2_vec = vld1q(src2);
            res_vec = vsubq(src1_vec, src2_vec);
            vst1q(ptr, res_vec);
        }
        if (ptr != dest_end)
        {
            generic::sub<T>(ptr, dest_end, src1, src2);
        }
    }
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    if constexpr (!((type::vector_lane<T> > 2)))
    {
        return generic::mul<T>(dest, dest_end, src1, src2);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
        constexpr size_t alignment = detail::get_recommended_alignment();
        detail::check_alignment(dest, alignment, "mul dest");
        detail::check_alignment(src1, alignment, "mul src1");
        detail::check_alignment(src2, alignment, "mul src2");
#endif

        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q(src1);
            src2_vec = vld1q(src2);
            res_vec = vmulq(src1_vec, src2_vec);
            vst1q(ptr, res_vec);
        }
        if (ptr != dest_end)
        {
            generic::mul<T>(ptr, dest_end, src1, src2);
        }
    }
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    if constexpr (!(std::is_floating_point_v<T>))
    {
        return generic::div<T>(dest, dest_end, src1, src2);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
        constexpr size_t alignment = detail::get_recommended_alignment();
        detail::check_alignment(dest, alignment, "div dest");
        detail::check_alignment(src1, alignment, "div src1");
        detail::check_alignment(src2, alignment, "div src2");
#endif

        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q(src1);
            src2_vec = vld1q(src2);
            res_vec = vdivq(src1_vec, src2_vec);
            vst1q(ptr, res_vec);
        }
        if (ptr != dest_end)
        {
            generic::div<T>(ptr, dest_end, src1, src2);
        }
    }
}

#else
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    return generic::check_between<T>(start, end, min_val, max_val);
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::add<T>(dest, dest_end, src1, src2);
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::sub<T>(dest, dest_end, src1, src2);
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::mul<T>(dest, dest_end, src1, src2);
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::div<T>(dest, dest_end, src1, src2);
}

#endif /* defined(__aarch64__) */

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */
