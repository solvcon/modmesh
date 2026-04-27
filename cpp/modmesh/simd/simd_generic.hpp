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

#include <concepts>
#include <functional>

namespace modmesh
{

namespace simd
{

namespace generic
{

template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    for (T const * ptr = start; ptr < end; ++ptr)
    {
        if (*ptr < min_val || *ptr > max_val)
        {
            return ptr;
        }
    }
    return nullptr;
}

template <typename T, std::invocable<T, T> ScalarOp>
inline void transform_binary(T * dest, T const * dest_end, T const * src1, T const * src2, ScalarOp scalar_op)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = scalar_op(*src1, *src2);
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T>
inline void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::plus<T>{});
}

template <typename T>
inline void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::minus<T>{});
}

template <typename T>
inline void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::multiplies<T>{});
}

template <typename T>
inline void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::divides<T>{});
}

template <typename T>
T max(T const * start, T const * end)
{
    T max_val = *start;
    for (T const * ptr = start + 1; ptr < end; ++ptr)
    {
        if (*ptr > max_val)
        {
            max_val = *ptr;
        }
    }
    return max_val;
}

} /* namespace generic */

} /* namespace simd */

} /* namespace modmesh */
