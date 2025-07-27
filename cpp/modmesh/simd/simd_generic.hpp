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

#include <algorithm>
#include <stdexcept>

namespace modmesh
{

namespace simd
{

namespace generic
{

template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    T const * ptr = start;
    while (ptr < end)
    {
        T idx = *ptr;
        if (idx < min_val || idx > max_val)
        {
            return ptr;
        }
        ++ptr;
    }
    return nullptr;
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = *src1 + *src2;
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = *src1 - *src2;
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = *src1 * *src2;
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = *src1 / *src2;
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T>
T * choose_pivot(T * left, T * right)
{
    T * first = left;
    T * mid = left + (right - left) / 2;
    T * last = right - 1;
    if (*mid < *first)
    {
        std::iter_swap(mid, first);
    }
    if (*last < *first)
    {
        std::iter_swap(last, first);
    }
    if (*last < *mid)
    {
        std::iter_swap(last, mid);
    }
    return mid;
}

template <typename T>
T * partition(T * left, T * right, T * pivot_pos)
{
    // 将pivot移到最右边
    std::iter_swap(pivot_pos, right - 1);
    T * store = left;

    // 将小于pivot的元素移到左边
    for (T * it = left; it < right - 1; ++it)
    {
        if (*it < *(right - 1))
        {
            std::iter_swap(it, store);
            ++store;
        }
    }

    // 将pivot放到正确位置
    std::iter_swap(store, right - 1);
    return store;
}

template <typename T>
T quick_select(T * left, T * right, size_t k)
{
    size_t len = right - left;
    if (k >= len)
    {
        throw std::out_of_range("quick_select: k out of range");
    }

    while (true)
    {
        T * pivot_it = choose_pivot(left, right);
        T * store = partition(left, right, pivot_it);
        size_t pivot_rank = store - left;

        if (pivot_rank == k)
        {
            return *store;
        }
        else if (pivot_rank < k)
        {
            k -= pivot_rank + 1;
            left = store + 1;
        }
        else
        {
            right = store;
        }
    }
}

template <typename T>
T median(T * dest, T * dest_end)
{
    const size_t n = dest_end - dest;
    if (n == 0)
    {
        throw std::runtime_error("median: empty array");
    }

    if (n & 1)
    {
        return quick_select(dest, dest_end, n / 2);
    }
    else
    {
        T v1 = quick_select(dest, dest_end, n / 2 - 1);
        T v2 = quick_select(dest, dest_end, n / 2);
        return static_cast<T>(v1 + v2) / static_cast<T>(2.0);
    }
}

} /* namespace generic */

} /* namespace simd */

} /* namespace modmesh */
