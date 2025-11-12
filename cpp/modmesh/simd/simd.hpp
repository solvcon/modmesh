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
#include <modmesh/simd/simd_support.hpp>

#include <modmesh/simd/neon/neon.hpp>

namespace modmesh
{

namespace simd
{

namespace detail
{
#ifndef NDEBUG
template <typename T>
inline bool is_aligned(T const * pointer, size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
}

template <typename T>
inline void check_alignment(T const * pointer, size_t required_alignment, const char * name)
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
#endif

// Get the recommended memory alignment for SIMD operations based on the detected SIMD instruction set.
inline constexpr size_t get_recommended_alignment()
{
#if defined(__aarch64__) || defined(__arm__)
    return 16;
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

} // namespace detail

// Check if each element from start to end (excluded end) is within the range [min_val, max_val)
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    using namespace detail;
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::check_between<T>(start, end, min_val, max_val);
        break;

    default:
        return generic::check_between<T>(start, end, min_val, max_val);
    }
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail;
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::add<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::add<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail;
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::sub<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::sub<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail;
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::mul<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::mul<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    using namespace detail;
    switch (detect_simd())
    {
    case SIMD_NEON:
        return neon::div<T>(dest, dest_end, src1, src2);
        break;

    default:
        return generic::div<T>(dest, dest_end, src1, src2);
    }
}

template <typename T>
T max(T const * start, T const * end)
{
    return generic::max<T>(start, end);
}

} /* namespace simd */

} /* namespace modmesh */
