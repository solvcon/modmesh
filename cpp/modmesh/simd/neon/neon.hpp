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
#include <algorithm>
#include <stdexcept>

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
        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q<T>(src1);
            src2_vec = vld1q<T>(src2);
            res_vec = vaddq<T>(src1_vec, src2_vec);
            vst1q<T>(ptr, res_vec);
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
        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q<T>(src1);
            src2_vec = vld1q<T>(src2);
            res_vec = vsubq<T>(src1_vec, src2_vec);
            vst1q<T>(ptr, res_vec);
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
        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q<T>(src1);
            src2_vec = vld1q<T>(src2);
            res_vec = vmulq<T>(src1_vec, src2_vec);
            vst1q<T>(ptr, res_vec);
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
        vec_t src1_vec;
        vec_t src2_vec;
        vec_t res_vec;
        T * ptr = dest;
        for (; ptr <= dest_end - N_lane; ptr += N_lane, src1 += N_lane, src2 += N_lane)
        {
            src1_vec = vld1q<T>(src1);
            src2_vec = vld1q<T>(src2);
            res_vec = vdivq<T>(src1_vec, src2_vec);
            vst1q<T>(ptr, res_vec);
        }
        if (ptr != dest_end)
        {
            generic::div<T>(ptr, dest_end, src1, src2);
        }
    }
}

/**
 * @brief 使用SIMD加速的三数取中法选择pivot
 *
 * 这个函数实现了快速排序中的pivot选择策略，使用NEON SIMD指令加速比较操作。
 * 三数取中法通过比较数组的首、中、尾三个元素，选择中间值作为pivot，
 * 这样可以避免在已排序数组上的最坏情况性能。
 *
 * @param left 数组左边界指针
 * @param right 数组右边界指针（指向最后一个元素的下一个位置）
 * @return 选择的pivot位置指针
 */
template <typename T>
T * choose_pivot_simd(T * left, T * right)
{
    // 选择三个位置：首、中、尾
    T * first = left; // 第一个元素
    T * mid = left + (right - left) / 2; // 中间元素
    T * last = right - 1; // 最后一个元素

    // 检查类型是否支持SIMD向量化
    if constexpr (type::has_vectype<T>)
    {
        using vec_t = type::vector_t<T>; // SIMD向量类型
        constexpr size_t N_lane = type::vector_lane<T>; // 向量中的元素数量

        // 只有当数组足够大时才使用SIMD，避免小数组的开销
        if (right - left >= N_lane * 2)
        {
            // 将三个值广播到SIMD向量中，每个向量元素都是相同的值
            vec_t first_vec = vdupq<T>(*first); // 将first值复制到向量的每个位置
            vec_t mid_vec = vdupq<T>(*mid); // 将mid值复制到向量的每个位置
            vec_t last_vec = vdupq<T>(*last); // 将last值复制到向量的每个位置

            // 使用SIMD比较：mid < first
            // vcltq<T> 执行向量化的"小于"比较，返回布尔掩码
            auto cmp_mid_first = vcltq<T>(mid_vec, first_vec);
            // 检查比较结果：vgetq提取向量的第0和第1个64位元素
            // 如果任一位置为真，说明mid < first
            if (vgetq<uint64_t, 0>(cmp_mid_first) || vgetq<uint64_t, 1>(cmp_mid_first))
            {
                std::iter_swap(mid, first); // 交换mid和first
            }

            // 使用SIMD比较：last < first
            auto cmp_last_first = vcltq<T>(last_vec, first_vec);
            if (vgetq<uint64_t, 0>(cmp_last_first) || vgetq<uint64_t, 1>(cmp_last_first))
            {
                std::iter_swap(last, first); // 交换last和first
            }

            // 使用SIMD比较：last < mid
            auto cmp_last_mid = vcltq<T>(last_vec, mid_vec);
            if (vgetq<uint64_t, 0>(cmp_last_mid) || vgetq<uint64_t, 1>(cmp_last_mid))
            {
                std::iter_swap(last, mid); // 交换last和mid
            }
        }
        else
        {
            // 对于小数组，使用标准的标量比较操作
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
        }
    }
    else
    {
        // 对于不支持SIMD的类型，使用标准的标量比较操作
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
    }

    return mid; // 返回中间位置作为pivot
}

/**
 * @brief 使用SIMD加速的数组分区操作
 *
 * 这个函数实现了快速排序中的分区操作，将数组分为两部分：
 * 左边部分的所有元素都小于pivot，右边部分的所有元素都大于等于pivot。
 * 使用NEON SIMD指令加速比较和交换操作。
 *
 * @param left 数组左边界指针
 * @param right 数组右边界指针
 * @param pivot_pos pivot元素的位置指针
 * @return pivot元素的最终位置指针
 */
template <typename T>
T * partition_simd(T * left, T * right, T * pivot_pos)
{
    // 将pivot元素移动到数组的最右边，便于后续处理
    std::iter_swap(pivot_pos, right - 1);
    T * store = left; // store指针指向下一个小于pivot的元素应该放置的位置

    // 检查类型是否支持SIMD向量化
    if constexpr (type::has_vectype<T>)
    {
        using vec_t = type::vector_t<T>; // SIMD向量类型
        constexpr size_t N_lane = type::vector_lane<T>; // 向量中的元素数量
        T pivot_val = *(right - 1); // 获取pivot值
        vec_t pivot_vec = vdupq<T>(pivot_val); // 将pivot值广播到SIMD向量

        // SIMD加速的分区操作：每次处理N_lane个元素
        T * it = left; // 当前处理位置
        T * simd_end = right - 1 - N_lane; // SIMD处理的结束位置

        // 使用SIMD批量处理数组元素
        for (; it <= simd_end; it += N_lane)
        {
            // vld1q<T> 从内存加载N_lane个元素到SIMD向量
            vec_t data_vec = vld1q<T>(it);
            // vcltq<T> 执行向量化的"小于"比较：data_vec < pivot_vec
            // 返回布尔掩码，每个元素表示对应位置是否小于pivot
            auto cmp_vec = vcltq<T>(data_vec, pivot_vec);

            // 将SIMD向量数据存储到临时数组，以便逐个处理
            T temp_data[N_lane];
            vst1q<T>(temp_data, data_vec); // 存储原始数据
            T temp_cmp[N_lane];
            vst1q<T>(temp_cmp, cmp_vec); // 存储比较结果（布尔掩码）

            // 逐个处理SIMD向量中的每个元素
            for (size_t i = 0; i < N_lane; ++i)
            {
                if (temp_cmp[i]) // 如果当前元素小于pivot
                {
                    // 将当前元素交换到store位置，并移动store指针
                    std::iter_swap(it + i, store);
                    ++store;
                }
            }
        }

        // 处理剩余的元素（无法用SIMD批量处理的部分）
        for (; it < right - 1; ++it)
        {
            if (*it < pivot_val)
            {
                std::iter_swap(it, store);
                ++store;
            }
        }
    }
    else
    {
        // 对于不支持SIMD的类型，使用标准的标量分区算法
        for (T * it = left; it < right - 1; ++it)
        {
            if (*it < *(right - 1))
            {
                std::iter_swap(it, store);
                ++store;
            }
        }
    }

    // 将pivot元素放到正确的位置（store指针位置）
    std::iter_swap(store, right - 1);
    return store; // 返回pivot的最终位置
}

/**
 * @brief 使用SIMD加速的快速选择算法
 *
 * 这个函数实现了快速选择算法，用于在未排序数组中查找第k小的元素。
 * 算法基于快速排序的分区思想，但只递归处理包含目标元素的那一半数组，
 * 平均时间复杂度为O(n)，最坏情况为O(n²)。
 * 使用NEON SIMD指令加速pivot选择和分区操作。
 *
 * @param left 数组左边界指针
 * @param right 数组右边界指针（指向最后一个元素的下一个位置）
 * @param k 要查找的元素排名（0-based，0表示最小元素）
 * @return 第k小的元素值
 * @throws std::out_of_range 当k超出数组范围时抛出异常
 */
template <typename T>
T quick_select_simd(T * left, T * right, size_t k)
{
    size_t len = right - left; // 计算数组长度
    if (k >= len)
    {
        throw std::out_of_range("quick_select_simd: k out of range");
    }

    // 主循环：不断分区直到找到第k小的元素
    while (true)
    {
        // 使用SIMD加速的三数取中法选择pivot
        // T * pivot_it = choose_pivot_simd<T>(left, right);
        T * pivot_it = generic::choose_pivot<T>(left, right);
        // 使用SIMD加速的分区操作，将pivot放到正确位置
        T * store = partition_simd<T>(left, right, pivot_it);
        // 计算pivot在分区后的排名（相对于left的位置）
        size_t pivot_rank = store - left;

        if (pivot_rank == k)
        {
            // 找到目标元素：pivot正好是第k小的元素
            return *store;
        }
        else if (pivot_rank < k)
        {
            // pivot排名小于k，目标元素在右半部分
            k -= pivot_rank + 1; // 调整k值（减去左半部分和pivot的数量）
            left = store + 1; // 将搜索范围缩小到右半部分
        }
        else
        {
            // pivot排名大于k，目标元素在左半部分
            right = store; // 将搜索范围缩小到左半部分
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

    // 对于小数组，直接使用通用算法
    if (n < type::vector_lane<T> * 4)
    {
        return generic::median<T>(dest, dest_end);
    }

    // 对于大数组，使用SIMD加速的quick_select
    if (n & 1)
    {
        return quick_select_simd<T>(dest, dest_end, n / 2);
    }
    else
    {
        T v1 = quick_select_simd<T>(dest, dest_end, n / 2 - 1);
        T v2 = quick_select_simd<T>(dest, dest_end, n / 2);
        return static_cast<T>(v1 + v2) / static_cast<T>(2.0);
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

template <typename T>
T median(T * dest, T * dest_end)
{
    return generic::median<T>(dest, dest_end);
}

#endif /* defined(__aarch64__) */

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */
