#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/buffer/ConcreteBuffer.hpp>
#include <modmesh/math/math.hpp>
#include <modmesh/simd/simd.hpp>

// TODO: Solve circular include between <modmesh/toggle/toggle.hpp> and SimpleArray class.
// Since it will happen circulate include when using <modmesh/toggle/toggle.hpp>,
// I use <modmesh/toggle/RadixTree.hpp> instead.
#include <modmesh/toggle/RadixTree.hpp>

#include <limits>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <algorithm>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace modmesh
{

template <typename T>
class SimpleArray; // forward declaration

namespace detail
{

template <size_t D, typename S>
size_t buffer_offset_impl(S const &)
{
    return 0;
}

template <size_t D, typename S, typename Arg, typename... Args>
size_t buffer_offset_impl(S const & strides, Arg arg, Args... args)
{
    return arg * strides[D] + buffer_offset_impl<D + 1>(strides, args...);
}

} /* end namespace detail */

template <typename S, typename... Args>
size_t buffer_offset(S const & strides, Args... args)
{
    return detail::buffer_offset_impl<0>(strides, args...);
}

inline size_t buffer_offset(small_vector<size_t> const & stride, small_vector<size_t> const & idx)
{
    if (stride.size() != idx.size())
    {
        throw std::out_of_range(Formatter() << "stride size " << stride.size() << " != "
                                            << "index size " << idx.size());
    }
    size_t offset = 0;
    for (size_t it = 0; it < stride.size(); ++it)
    {
        offset += stride[it] * idx[it];
    }
    return offset;
}

namespace detail
{

using shape_type = small_vector<size_t>;
using sshape_type = small_vector<ssize_t>;
using slice_type = small_vector<ssize_t>;

template <typename T>
struct SimpleArrayInternalTypes
{
    using value_type = T;
    using shape_type = detail::shape_type;
    using sshape_type = detail::sshape_type;
    using buffer_type = ConcreteBuffer;
}; /* end class SimpleArrayInternalType */

template <typename A, typename T>
class SimpleArrayMixinModifiers
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    A & fill(value_type const & value)
    {
        auto athis = static_cast<A *>(this);
        std::fill(athis->begin(), athis->end(), value);
        return *athis;
    }

}; /* end class SimpleArrayMixinModifiers */

template <typename U>
struct select_real_t
{
    using type = U;
};

template <typename U>
struct select_real_t<Complex<U>>
{
    using type = U;
};

template <typename A, typename T>
class SimpleArrayMixinCalculators
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;
    using real_type = typename detail::select_real_t<value_type>::type;

    template <typename RedFn, typename... RedArgs>
    auto reduce(const shape_type & axis, RedFn red_fn, RedArgs &&... red_args) const
    {
        using element_type = std::invoke_result_t<
            RedFn,
            const A *,
            small_vector<value_type> &,
            RedArgs...>;

        using ret_type = typename A::template rebind<element_type>;

        auto athis = static_cast<const A *>(this);
        const size_t ndim = athis->ndim();

        small_vector<bool> reduce_mask(ndim, false);
        for (size_t ax : axis)
        {
            if (ax >= ndim || ax < 0)
            {
                throw std::out_of_range("reduce: axis out of range");
            }
            reduce_mask[ax] = true;
        }

        size_t red_count = reduce_mask.count(true);
        if (red_count == 0 || red_count == ndim)
        {
            throw std::runtime_error("reduce: no axis to reduce or all axes are reduced");
        }

        small_vector<size_t> out_shape(ndim - red_count);
        for (size_t i = 0, l = 0; i < ndim; ++i)
        {
            if (!reduce_mask[i])
            {
                out_shape[l++] = athis->shape(i);
            }
        }
        ret_type result(out_shape);

        small_vector<size_t> red_axes(red_count), red_shape(red_count);
        for (size_t i = 0, l = 0; i < ndim; ++i)
        {
            if (reduce_mask[i])
            {
                red_axes[l] = i;
                red_shape[l++] = athis->shape(i);
            }
        }

        small_vector<size_t> full_idx(ndim, 0);
        auto out_idx = result.first_sidx();

        do
        {
            for (size_t i = 0, l = 0; i < ndim; ++i)
            {
                if (!reduce_mask[i])
                {
                    full_idx[i] = out_idx[l++];
                }
            }

            small_vector<value_type> slice;
            small_vector<size_t> red_idx(red_axes.size(), 0);

            do
            {
                for (size_t k = 0; k < red_axes.size(); ++k)
                {
                    full_idx[red_axes[k]] = red_idx[k];
                }
                slice.push_back(athis->at(full_idx));

            } while (red_idx.next_cartesian_product(red_shape));

            element_type mv = std::invoke(red_fn, this, slice, std::forward<RedArgs>(red_args)...);
            result.at(out_idx) = mv;
        } while (result.next_sidx(out_idx));

        return result;
    }

    value_type median_op(small_vector<value_type> & sv) const;

    value_type median_freq(small_vector<value_type> & sv) const;

    A median(const small_vector<size_t> & axis) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::median_op);
    }

    value_type median() const
    {
        MODMESH_PROFILE_SCOPE("SimpleArray::median()");
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        small_vector<T> acopy(n);
        auto sidx = athis->first_sidx();
        size_t i = 0;
        do
        {
            acopy[i] = athis->at(sidx);
            ++i;
        } while (athis->next_sidx(sidx));
        return median_op(acopy);
    }

    value_type average_op(small_vector<value_type> & sv, small_vector<value_type> & weight) const
    {
        const size_t n = sv.size();
        if (n != weight.size())
        {
            throw std::runtime_error("SimpleArray::average_op(): weight size does not match array size");
        }
        value_type sum = 0;
        value_type total_weight = 0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += sv[i] * weight[i];
            total_weight += weight[i];
        }
        if (total_weight == static_cast<value_type>(0))
        {
            throw std::runtime_error("SimpleArray::average_op(): total weight is zero");
        }
        return sum / total_weight;
    }

    A average(const small_vector<size_t> & axis, A const & weight) const
    {
        small_vector<value_type> weight_sv(weight.size());
        auto sidx = weight.first_sidx();
        size_t i = 0;
        do
        {
            weight_sv[i] = weight.at(sidx);
            ++i;
        } while (weight.next_sidx(sidx));
        return reduce(axis, &SimpleArrayMixinCalculators::average_op, weight_sv);
    }

    value_type average(A const & weight) const
    {
        auto athis = static_cast<A const *>(this);
        const shape_type & weight_shape = weight.shape();
        const shape_type & athis_shape = athis->shape();
        for (size_t i = 0; i < athis_shape.size(); ++i)
        {
            if (weight_shape[i] != athis_shape[i])
            {
                throw std::runtime_error("SimpleArray::average(): weight shape does not match array shape");
            }
        }
        value_type sum = 0;
        value_type total_weight = 0;
        auto sidx = athis->first_sidx();
        do
        {
            sum += athis->at(sidx) * weight.at(sidx);
            total_weight += weight.at(sidx);
        } while (athis->next_sidx(sidx));
        if (total_weight == static_cast<value_type>(0))
        {
            throw std::runtime_error("SimpleArray::average(): total weight is zero");
        }
        return sum / total_weight;
    }

    value_type mean_op(small_vector<value_type> & sv) const
    {
        const size_t n = sv.size();
        value_type sum = 0;
        for (const auto & v : sv)
        {
            sum += v;
        }
        return sum / static_cast<value_type>(n);
    }

    A mean(const small_vector<size_t> & axis) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::mean_op);
    }

    value_type mean() const
    {
        auto athis = static_cast<A const *>(this);
        auto sidx = athis->first_sidx();
        value_type sum = 0;
        int64_t total = 0;
        do
        {
            sum += athis->at(sidx);
            ++total;
        } while (athis->next_sidx(sidx));
        return sum / static_cast<value_type>(total);
    }

    real_type var_op(small_vector<value_type> & sv, size_t ddof) const
    {
        const size_t n = sv.size();
        if (n <= ddof)
        {
            throw std::runtime_error("SimpleArray::var_op(): ddof must be less than the number of elements");
        }
        value_type mu = mean_op(sv);
        real_type acc = 0;
        if constexpr (is_complex_v<value_type>)
        {
            for (const auto & v : sv)
            {
                acc += (v - mu).norm();
            }
        }
        else
        {
            for (const auto & v : sv)
            {
                acc += (v - mu) * (v - mu);
            }
        }
        return acc / static_cast<real_type>(n - ddof);
    }

    auto var(const small_vector<size_t> & axis, size_t ddof) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::var_op, ddof);
    }

    real_type var(size_t ddof) const
    {
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        if (n <= ddof)
        {
            throw std::runtime_error("SimpleArray::var(): ddof must be less than the number of elements");
        }

        auto sidx = athis->first_sidx();
        value_type mu = athis->mean();
        real_type acc = 0;
        if constexpr (is_complex_v<value_type>)
        {
            do
            {
                acc += athis->at(sidx).norm();
            } while (athis->next_sidx(sidx));
        }
        else
        {
            do {
                acc += athis->at(sidx) * athis->at(sidx);
            } while (athis->next_sidx(sidx));
        }
        if constexpr (is_complex_v<value_type>)
        {
            acc -= n * mu.norm();
        }
        else
        {
            acc -= n * mu * mu;
        }
        return acc / static_cast<real_type>(n - ddof);
    }

    real_type std_op(small_vector<value_type> & sv, size_t ddof) const
    {
        return std::sqrt(var_op(sv, ddof));
    }

    auto std(const small_vector<size_t> & axis, size_t ddof) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::std_op, ddof);
    }

    real_type std(size_t ddof) const
    {
        auto athis = static_cast<A const *>(this);
        return std::sqrt(athis->var(ddof));
    }

    value_type min() const
    {
        value_type initial = std::numeric_limits<value_type>::max();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) < initial)
            {
                initial = athis->data(i);
            }
        }
        return initial;
    }

    value_type max() const
    {
        value_type initial = std::numeric_limits<value_type>::lowest();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) > initial)
            {
                initial = athis->data(i);
            }
        }
        return initial;
    }

    value_type sum() const
    {
        value_type initial;
        if constexpr (is_complex_v<value_type>)
        {
            initial = value_type();
        }
        else
        {
            initial = 0;
        }

        auto athis = static_cast<A const *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            for (size_t i = 0; i < athis->size(); ++i)
            {
                initial += athis->data(i);
            }
        }
        else
        {
            for (size_t i = 0; i < athis->size(); ++i)
            {
                initial |= athis->data(i);
            }
        }
        return initial;
    }

    A abs() const
    {
        auto athis = static_cast<A const *>(this);
        A ret(*athis);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>> && std::is_signed_v<value_type>)
        {
            for (size_t i = 0; i < athis->size(); ++i)
            {
                ret.data(i) = std::abs(athis->data(i));
            }
        }
        return ret;
    }

    A add(A const & other) const
    {
        return A(*static_cast<A const *>(this)).iadd(other);
    }

    A sub(A const & other) const
    {
        return A(*static_cast<A const *>(this)).isub(other);
    }

    A mul(A const & other) const
    {
        return A(*static_cast<A const *>(this)).imul(other);
    }

    A div(A const & other) const
    {
        return A(*static_cast<A const *>(this)).idiv(other);
    }

    A & iadd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr += *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            size_t size = athis->size();
            for (size_t i = 0; i < size; ++i)
            {
                athis->data(i) = athis->data(i) || other.data(i);
            }
        }

        return *athis;
    }

    A & isub(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr -= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            throw std::runtime_error(Formatter() << "SimpleArray<bool>::isub(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A & imul(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr *= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            size_t size = athis->size();
            for (size_t i = 0; i < size; ++i)
            {
                athis->data(i) = athis->data(i) && other.data(i);
            }
        }
        return *athis;
    }

    A & idiv(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr /= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            throw std::runtime_error(Formatter() << "SimpleArray<bool>::idiv(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A add_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::add<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A sub_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::sub<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A mul_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::mul<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A div_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::div<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A & iadd_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::add<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->iadd(other);
        }
    }

    A & isub_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::sub<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->isub(other);
        }
    }

    A & imul_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::mul<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->imul(other);
        }
    }

    A & idiv_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::div<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->idiv(other);
        }
    }

    A matmul(A const & other) const;

    A & imatmul(A const & other);

private:
    static void find_two_bins(const uint32_t * freq, size_t n, int & bin1, int & bin2);
}; /* end class SimpleArrayMixinCalculators */

template <typename A, typename T>
typename detail::SimpleArrayMixinCalculators<A, T>::value_type
detail::SimpleArrayMixinCalculators<A, T>::median_op(small_vector<value_type> & sv) const
{
    MODMESH_PROFILE_SCOPE("SimpleArray::median_op()");
    const size_t n = sv.size();

    if constexpr (std::is_same_v<value_type, int8_t> ||
                  std::is_same_v<value_type, uint8_t> ||
                  std::is_same_v<value_type, bool>)
    {
        return median_freq(sv);
    }
    else
    {
        value_type v1 = sv.select_kth(n / 2);
        if (n % 2 != 0)
        {
            return v1;
        }
        value_type v2 = simd::max(sv.begin(), sv.begin() + n / 2);
        value_type result = static_cast<value_type>(v1 + v2) / static_cast<value_type>(2.0);
        return result;
    }
}

/**
 * Calculate median using frequency counting for small data types.
 * This algorithm is optimized for uint8_t, int8_t, and bool types where
 * the range of possible values is small (≤256). Instead of sorting,
 * it counts the frequency of each value and finds the median position.
 *
 * Algorithm:
 * 1. Count frequency of each possible value (0-255 for uint8, -128-127 for int8, 0-1 for bool)
 * 2. Find the two bins that contain the median elements using cumulative frequency
 * 3. Calculate median based on the positions found
 *
 * Time complexity: O(n)
 * Space complexity: O(k) for the frequency array (k = 256 for 8-bit types)
 */
template <typename A, typename T>
typename detail::SimpleArrayMixinCalculators<A, T>::value_type
detail::SimpleArrayMixinCalculators<A, T>::median_freq(small_vector<value_type> & sv) const
{
    MODMESH_PROFILE_SCOPE("SimpleArray::median_freq()");

    const size_t n = sv.size();
    if (n == 0)
    {
        throw std::runtime_error("median_freq(): empty container");
    }

    uint32_t freq[256] = {};

    if constexpr (std::is_same_v<value_type, uint8_t>)
    {
        for (uint8_t v : sv) { ++freq[v]; }
    }
    else if constexpr (std::is_same_v<value_type, int8_t>)
    {
        for (int8_t v : sv) { ++freq[static_cast<unsigned>(static_cast<int>(v) + 128)]; }
    }
    else
    {
        for (bool v : sv) { ++freq[v ? 1 : 0]; }
    }

    int b1, b2;
    find_two_bins(freq, n, b1, b2);

    if constexpr (std::is_same_v<value_type, uint8_t>)
    {
        const int m = (b1 + b2) / 2;
        return static_cast<value_type>(m);
    }
    else if constexpr (std::is_same_v<value_type, int8_t>)
    {
        const int v1 = b1 - 128;
        const int v2 = b2 - 128;
        return static_cast<value_type>((v1 + v2) / 2);
    }
    else
    {
        const uint32_t ones = freq[1];
        return static_cast<value_type>(ones * 2 >= n);
    }
}

/**
 * Perform matrix multiplication for 2D arrays.
 * This implementation supports only 2D × 2D matrix multiplication.
 */
template <typename A, typename T>
A SimpleArrayMixinCalculators<A, T>::matmul(A const & other) const
{
    auto athis = static_cast<A const *>(this);
    const size_t this_ndim = athis->ndim();
    const size_t other_ndim = other.ndim();

    auto format_shape = [](A const * arr) -> std::string
    {
        Formatter shape_formatter;
        if (arr->ndim() == 0)
        {
            shape_formatter << "()";
        }
        else
        {
            shape_formatter << "(";
            for (size_t i = 0; i < arr->ndim(); ++i)
            {
                if (i > 0)
                    shape_formatter << ",";
                shape_formatter << arr->shape(i);
            }
            shape_formatter << ")";
        }
        return shape_formatter.str();
    };

    if (this_ndim != 2 || other_ndim != 2)
    {
        throw std::out_of_range(Formatter() << "SimpleArray::matmul(): unsupported dimensions: this="
                                            << format_shape(athis) << " other=" << format_shape(&other)
                                            << ". Only 2D x 2D matrix multiplication is supported");
    }

    const size_t m = athis->shape(0);
    const size_t k = athis->shape(1);
    const size_t n = other.shape(1);

    if (k != other.shape(0))
    {
        throw std::out_of_range(Formatter() << "SimpleArray::matmul(): shape mismatch: this="
                                            << format_shape(athis) << " other=" << format_shape(&other));
    }

    typename detail::SimpleArrayInternalTypes<T>::shape_type result_shape{m, n};
    A result(result_shape);
    result.fill(static_cast<value_type>(0));

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t l = 0; l < k; ++l)
            {
                result(i, j) += athis->operator()(i, l) * other(l, j);
            }
        }
    }

    return result;
}

/**
 * Perform in-place matrix multiplication for 2D arrays.
 * This implementation supports only 2D × 2D matrix multiplication.
 * The result replaces the content of the current array.
 */
template <typename A, typename T>
A & SimpleArrayMixinCalculators<A, T>::imatmul(A const & other)
{
    auto athis = static_cast<A *>(this);
    A result = athis->matmul(other);
    *athis = std::move(result);

    return *athis;
}

/**
 * Find the two bins that correspond to the median values for frequency-based median calculation.
 * This function is used for small data types (uint8_t, int8_t, bool) where frequency counting
 * is more efficient than sorting.
 */
template <typename A, typename T>
void SimpleArrayMixinCalculators<A, T>::find_two_bins(const uint32_t * freq, size_t n, int & bin1, int & bin2)
{
    const size_t k1 = (n - 1) / 2;
    const size_t k2 = n / 2;

    uint32_t cumulative = 0;
    bin1 = -1;
    bin2 = -1;

    for (int i = 0; i < 256; ++i)
    {
        cumulative += freq[i];
        if (bin1 < 0 && cumulative > k1)
        {
            bin1 = i;
        }
        if (bin2 < 0 && cumulative > k2)
        {
            bin2 = i;
        }
        if (bin1 >= 0 && bin2 >= 0)
        {
            break;
        }
    }
}

template <typename A, typename T>
class SimpleArrayMixinSort
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    void sort(void);
    SimpleArray<uint64_t> argsort(void);
    template <typename I>
    A take_along_axis(SimpleArray<I> const & indices);
    template <typename I>
    A take_along_axis_simd(SimpleArray<I> const & indices);

}; /* end class SimpleArrayMixinSort */

template <typename A, typename T>
void SimpleArrayMixinSort<A, T>::sort(void)
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(Formatter() << "SimpleArray::sort(): currently only support 1D array but the array is "
                                             << athis->ndim() << " dimension");
    }

    std::sort(athis->begin(), athis->end());
}

template <typename T, typename I>
void indexed_copy(T * dest, T const * data, I const * begin, I const * const end);

template <typename T>
T const * check_index_range(SimpleArray<T> const & indices, size_t max_idx);

template <typename A, typename T>
class SimpleArrayMixinSearch
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    size_t argmin() const
    {
        size_t min_index = 0;
        value_type min_value = std::numeric_limits<value_type>::max();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) < min_value)
            {
                min_value = athis->data(i);
                min_index = i;
            }
        }
        return min_index;
    }

    size_t argmax() const
    {
        size_t max_index = 0;
        value_type max_value = std::numeric_limits<value_type>::lowest();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) > max_value)
            {
                max_value = athis->data(i);
                max_index = i;
            }
        }
        return max_index;
    }
}; /* end class SimpleArrayMixinSearch */

} /* end namespace detail */

/**
 * Simple array type for contiguous memory storage. Size does not change. The
 * copy semantics performs data copy. The move semantics invalidates the
 * existing memory buffer.
 */
template <typename T>
class SimpleArray
    : public detail::SimpleArrayMixinModifiers<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinCalculators<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinSort<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinSearch<SimpleArray<T>, T>
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:
    template <typename U>
    using rebind = SimpleArray<U>;
    using value_type = typename internal_types::value_type;
    using shape_type = typename internal_types::shape_type;
    using sshape_type = typename internal_types::sshape_type;
    using buffer_type = typename internal_types::buffer_type;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    static constexpr size_t itemsize() { return ITEMSIZE; }

    explicit SimpleArray(size_t length)
        : m_buffer(buffer_type::construct(length * ITEMSIZE))
        , m_shape{length}
        , m_stride{1}
        , m_body(m_buffer->template data<T>())
    {
    }

    template <class InputIt>
    SimpleArray(InputIt first, InputIt last)
        : SimpleArray(last - first)
    {
        std::copy(first, last, data());
    }

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit SimpleArray(small_vector<size_t> const & shape)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE);
            m_body = m_buffer->template data<T>();
        }
    }

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit SimpleArray(small_vector<size_t> const & shape, value_type const & value)
        : SimpleArray(shape)
    {
        std::fill(begin(), end(), value);
    }

    explicit SimpleArray(std::vector<size_t> const & shape)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE);
            m_body = m_buffer->template data<T>();
        }
    }

    explicit SimpleArray(std::vector<size_t> const & shape, value_type const & value)
        : SimpleArray(shape)
    {
        std::fill(begin(), end(), value);
    }

    explicit SimpleArray(std::shared_ptr<buffer_type> const & buffer)
    {
        if (buffer)
        {
            const size_t nitem = buffer->nbytes() / ITEMSIZE;
            if (buffer->nbytes() != nitem * ITEMSIZE)
            {
                throw std::runtime_error("SimpleArray: input buffer size must be divisible");
            }
            m_shape = shape_type{nitem};
            m_stride = shape_type{1};
            m_buffer = buffer;
            m_body = m_buffer->template data<T>();
        }
        else
        {
            throw std::runtime_error("SimpleArray: buffer cannot be null");
        }
    }

    explicit SimpleArray(small_vector<size_t> const & shape, std::shared_ptr<buffer_type> const & buffer)
        : SimpleArray(buffer)
    {
        if (buffer)
        {
            m_shape = shape;
            m_stride = calc_stride(m_shape);
            const size_t nbytes = m_shape[0] * m_stride[0] * ITEMSIZE;
            if (nbytes != buffer->nbytes())
            {
                throw std::runtime_error(Formatter() << "SimpleArray: shape byte count " << nbytes
                                                     << " differs from buffer " << buffer->nbytes());
            }
        }
    }

    explicit SimpleArray(small_vector<size_t> const & shape,
                         small_vector<size_t> const & stride,
                         std::shared_ptr<buffer_type> const & buffer,
                         bool c_contiguous,
                         bool f_contiguous)
        : SimpleArray(shape, stride, buffer)
    {
        if (shape.size() != stride.size())
        {
            throw std::runtime_error("SimpleArray: shape and stride size mismatch");
        }
        if (c_contiguous)
        {
            check_c_contiguous(shape, stride);
        }
        if (f_contiguous)
        {
            check_f_contiguous(shape, stride);
        }
    }

    explicit SimpleArray(small_vector<size_t> const & shape,
                         small_vector<size_t> const & stride,
                         std::shared_ptr<buffer_type> const & buffer)
        : SimpleArray(buffer)
    {
        if (buffer)
        {
            m_shape = shape;
            m_stride = stride;
        }
    }

    SimpleArray(std::initializer_list<T> init)
        : SimpleArray(init.size())
    {
        std::copy_n(init.begin(), init.size(), data());
    }

    SimpleArray()
        : SimpleArray(0)
    {
    }

    SimpleArray(SimpleArray const & other)
        : m_buffer(other.m_buffer->clone())
        , m_shape(other.m_shape)
        , m_stride(other.m_stride)
        , m_nghost(other.m_nghost)
        , m_body(calc_body(m_buffer->template data<T>(), m_stride, other.m_nghost))
    {
    }

    SimpleArray(SimpleArray && other) noexcept
        : m_buffer(std::move(other.m_buffer))
        , m_shape(std::move(other.m_shape))
        , m_stride(std::move(other.m_stride))
        , m_nghost(other.m_nghost)
        , m_body(other.m_body)
    {
    }

    SimpleArray & operator=(SimpleArray const & other)
    {
        if (this != &other)
        {
            *m_buffer = *(other.m_buffer); // Size is checked inside.
            m_shape = other.m_shape;
            m_stride = other.m_stride;
            m_nghost = other.m_nghost;
            m_body = calc_body(m_buffer->template data<T>(), m_stride, other.m_nghost);
        }
        return *this;
    }

    SimpleArray & operator=(SimpleArray && other) noexcept
    {
        if (this != &other)
        {
            m_buffer = std::move(other.m_buffer);
            m_shape = std::move(other.m_shape);
            m_stride = std::move(other.m_stride);
            m_nghost = other.m_nghost;
            m_body = other.m_body;
        }
        return *this;
    }

    ~SimpleArray() = default;

    template <typename... Args>
    SimpleArray & remake(Args &&... args)
    {
        SimpleArray(args...).swap(*this);
        return *this;
    }

    static shape_type calc_stride(shape_type const & shape)
    {
        shape_type stride(shape.size());
        if (!shape.empty())
        {
            stride[shape.size() - 1] = 1;
            for (size_t it = shape.size() - 1; it > 0; --it)
            {
                stride[it - 1] = stride[it] * shape[it];
            }
        }
        return stride;
    }

    static T * calc_body(T * data, shape_type const & stride, size_t nghost)
    {
        if (nullptr == data || stride.empty() || 0 == nghost)
        {
            // Do nothing.
        }
        else
        {
            shape_type shape(stride.size(), 0);
            shape[0] = nghost;
            data += buffer_offset(stride, shape);
        }
        return data;
    }

    /**
     * Create an identity matrix of size n x n.
     *
     * @param n Size of the square identity matrix
     * @return SimpleArray representing an n x n identity matrix
     */
    static SimpleArray eye(size_t n)
    {
        shape_type shape{n, n};
        SimpleArray result(shape, static_cast<value_type>(0));

        // Set diagonal elements to 1
        for (size_t i = 0; i < n; ++i)
        {
            result(i, i) = static_cast<value_type>(1);
        }

        return result;
    }

    explicit operator bool() const noexcept { return bool(m_buffer) && bool(*m_buffer); }

    size_t nbytes() const noexcept { return size() * ITEMSIZE; }
    /**
     * In the following numpy documentaions,
     * `size()` are calculated based on the number of the elements.
     * https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.size.html
     * SimpleArray matchs the behavior of numpy.ndarray.
     */
    size_t size() const noexcept
    {
        size_t size = 1;
        for (size_t it = 0; it < m_shape.size(); ++it)
        {
            size *= m_shape[it];
        }
        return size;
    }

    using iterator = T *;
    using const_iterator = T const *;

    iterator begin() noexcept { return data(); }
    iterator end() noexcept { return data() + size(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end() const noexcept { return data() + size(); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    value_type const & operator[](size_t it) const noexcept { return data(it); }
    value_type & operator[](size_t it) noexcept { return data(it); }

    value_type const & at(size_t it) const
    {
        validate_range(it);
        return data(it);
    }
    value_type & at(size_t it)
    {
        validate_range(it);
        return data(it);
    }

    value_type const & at(ssize_t it) const
    {
        validate_range(it);
        it += m_nghost;
        return data(it);
    }
    value_type & at(ssize_t it)
    {
        validate_range(it);
        it += m_nghost;
        return data(it);
    }

    value_type const & at(std::vector<size_t> const & idx) const { return at(shape_type(idx)); }
    value_type & at(std::vector<size_t> const & idx) { return at(shape_type(idx)); }

    value_type const & at(shape_type const & idx) const
    {
        const size_t offset = buffer_offset(m_stride, idx);
        validate_range(offset);
        return data(offset);
    }
    value_type & at(shape_type const & idx)
    {
        const size_t offset = buffer_offset(m_stride, idx);
        validate_range(offset);
        return data(offset);
    }

    value_type const & at(std::vector<ssize_t> const & idx) const { return at(sshape_type(idx)); }
    value_type & at(std::vector<ssize_t> const & idx) { return at(sshape_type(idx)); }

    value_type const & at(sshape_type sidx) const
    {
        validate_shape(sidx);
        sidx[0] += m_nghost;
        shape_type const idx(sidx.begin(), sidx.end());
        const size_t offset = buffer_offset(m_stride, idx);
        return data(offset);
    }
    value_type & at(sshape_type sidx)
    {
        validate_shape(sidx);
        sidx[0] += m_nghost;
        shape_type const idx(sidx.begin(), sidx.end());
        const size_t offset = buffer_offset(m_stride, idx);
        return data(offset);
    }

    shape_type first_sidx() const noexcept
    {
        return shape_type(shape().size(), 0);
    }

    bool next_sidx(shape_type & sidx) const noexcept
    {
        return sidx.next_cartesian_product(shape());
    }

    size_t ndim() const noexcept { return m_shape.size(); }
    shape_type const & shape() const { return m_shape; }
    size_t shape(size_t it) const noexcept { return m_shape[it]; }
    size_t & shape(size_t it) noexcept { return m_shape[it]; }
    shape_type const & stride() const { return m_stride; }
    size_t stride(size_t it) const noexcept { return m_stride[it]; }
    size_t & stride(size_t it) noexcept { return m_stride[it]; }

    size_t nghost() const { return m_nghost; }
    size_t nbody() const { return m_shape.empty() ? 0 : m_shape[0] - m_nghost; }
    bool has_ghost() const { return m_nghost != 0; }
    void set_nghost(size_t nghost)
    {
        if (0 != nghost)
        {
            if (0 == ndim())
            {
                throw std::out_of_range(
                    Formatter() << "SimpleArray: cannot set nghost " << nghost << " > 0 to an empty array");
            }
            if (nghost > shape(0))
            {
                throw std::out_of_range(
                    Formatter() << "SimpleArray: cannot set nghost " << nghost << " > shape(0) " << shape(0));
            }
        }
        m_nghost = nghost;
        if (bool(*this))
        {
            m_body = calc_body(m_buffer->template data<T>(), m_stride, m_nghost);
        }
    }

    template <typename U>
    SimpleArray<U> reshape(shape_type const & shape) const
    {
        return SimpleArray<U>(shape, m_buffer);
    }

    SimpleArray reshape(shape_type const & shape) const
    {
        return SimpleArray(shape, m_buffer);
    }

    SimpleArray reshape() const
    {
        return SimpleArray(m_shape, m_buffer);
    }

    void swap(SimpleArray & other) noexcept
    {
        if (this != &other)
        {
            std::swap(m_buffer, other.m_buffer);
            std::swap(m_shape, other.m_shape);
            std::swap(m_stride, other.m_stride);
            std::swap(m_nghost, other.m_nghost);
            std::swap(m_body, other.m_body);
        }
    }

    void transpose()
    {
        std::reverse(m_shape.begin(), m_shape.end());
        std::reverse(m_stride.begin(), m_stride.end());
    }

    void transpose(shape_type const & axis)
    {
        if (axis.size() != m_shape.size())
        {
            throw std::runtime_error("SimpleArray: axis size mismatch");
        }
        shape_type new_shape(m_shape.size(), -1);
        shape_type new_stride(m_stride.size());
        for (size_t it = 0; it < m_shape.size(); ++it)
        {
            if (axis[it] >= m_shape.size() || axis[it] < 0)
            {
                throw std::runtime_error("SimpleArray: axis out of range");
            }
            if (new_shape[it] != -1)
            {
                throw std::runtime_error("SimpleArray: axis already set");
            }
            new_shape[it] = m_shape[axis[it]];
            new_stride[it] = m_stride[axis[it]];
        }
        m_shape = new_shape;
        m_stride = new_stride;
    }

    template <typename... Args>
    value_type const & operator()(Args... args) const { return *vptr(args...); }
    template <typename... Args>
    value_type & operator()(Args... args) { return *vptr(args...); }

    template <typename... Args>
    value_type const * vptr(Args... args) const { return m_body + buffer_offset(m_stride, args...); }
    template <typename... Args>
    value_type * vptr(Args... args) { return m_body + buffer_offset(m_stride, args...); }

    /* Backdoor */
    value_type const & data(size_t it) const { return data()[it]; }
    value_type & data(size_t it) { return data()[it]; }
    value_type const * data() const { return buffer().template data<value_type>(); }
    value_type * data() { return buffer().template data<value_type>(); }

    buffer_type const & buffer() const { return *m_buffer; }
    buffer_type & buffer() { return *m_buffer; }

    value_type const * body() const { return m_body; }
    value_type * body() { return m_body; }

private:
    void check_c_contiguous(small_vector<size_t> const & shape,
                            small_vector<size_t> const & stride) const
    {
        if (stride[stride.size() - 1] != 1)
        {
            throw std::runtime_error("SimpleArray: C contiguous stride must end with 1");
        }
        for (size_t it = 0; it < shape.size() - 1; ++it)
        {
            if (stride[it] != shape[it + 1] * stride[it + 1])
            {
                throw std::runtime_error("SimpleArray: C contiguous stride must match shape");
            }
        }
    }

    void check_f_contiguous(small_vector<size_t> const & shape,
                            small_vector<size_t> const & stride) const
    {
        if (stride[0] != 1)
        {
            throw std::runtime_error("SimpleArray: Fortran contiguous stride must start with 1");
        }
        for (size_t it = 0; it < shape.size() - 1; ++it)
        {
            if (stride[it + 1] != shape[it] * stride[it])
            {
                throw std::runtime_error("SimpleArray: Fortran contiguous stride must match shape");
            }
        }
    }

    void validate_range(ssize_t it) const
    {
        if (m_nghost != 0 && ndim() != 1)
        {
            throw std::out_of_range(
                Formatter() << "SimpleArray::validate_range(): cannot handle "
                            << ndim() << "-dimensional (more than 1) array with non-zero nghost: " << m_nghost);
        }
        if (it < -static_cast<ssize_t>(m_nghost))
        {
            throw std::out_of_range(Formatter() << "SimpleArray: index " << it << " < -nghost: " << -static_cast<ssize_t>(m_nghost));
        }
        if (it >= static_cast<ssize_t>((buffer().nbytes() / ITEMSIZE) - m_nghost))
        {
            throw std::out_of_range(
                Formatter() << "SimpleArray: index " << it << " >= " << (buffer().nbytes() / ITEMSIZE) - m_nghost
                            << " (buffer size: " << (buffer().nbytes() / ITEMSIZE) << " - nghost: " << m_nghost << ")");
        }
    }

    void validate_shape(small_vector<ssize_t> const & idx) const
    {
        auto index2string = [&idx]()
        {
            Formatter ms;
            ms << "[";
            for (size_t it = 0; it < idx.size(); ++it)
            {
                ms << idx[it];
                if (it != idx.size() - 1)
                {
                    ms << ", ";
                }
            }
            ms << "]";
            return ms.str();
        };

        // Test for the "index shape".
        if (idx.empty())
        {
            throw std::out_of_range("SimpleArray::validate_shape(): empty index");
        }
        if (idx.size() != m_shape.size())
        {
            throw std::out_of_range(Formatter() << "SimpleArray: dimension of input indices " << index2string()
                                                << " != array dimension " << m_shape.size());
        }

        // Test the first dimension.
        if (idx[0] < -static_cast<ssize_t>(m_nghost))
        {
            throw std::out_of_range(Formatter() << "SimpleArray: dim 0 in " << index2string()
                                                << " < -nghost: " << -static_cast<ssize_t>(m_nghost));
        }
        if (idx[0] >= static_cast<ssize_t>(nbody()))
        {
            throw std::out_of_range(Formatter() << "SimpleArray: dim 0 in " << index2string() << " >= nbody: " << nbody()
                                                << " (shape[0]: " << m_shape[0] << " - nghost: " << nghost() << ")");
        }

        // Test the rest of the dimensions.
        for (size_t it = 1; it < m_shape.size(); ++it)
        {
            if (idx[it] < 0)
            {
                throw std::out_of_range(Formatter() << "SimpleArray: dim " << it << " in " << index2string() << " < 0");
            }
            if (idx[it] >= static_cast<ssize_t>(m_shape[it]))
            {
                throw std::out_of_range(Formatter() << "SimpleArray: dim " << it << " in " << index2string()
                                                    << " >= shape[" << it << "]: " << m_shape[it]);
            }
        }
    }

    /// Contiguous data buffer for the array.
    std::shared_ptr<buffer_type> m_buffer;
    /// Each element in this vector is the number of element in the
    /// corresponding dimension.
    shape_type m_shape;
    /// Each element in this vector is the number of elements (not number of
    /// bytes) to skip for advancing an index in the corresponding dimension.
    shape_type m_stride;

    size_t m_nghost = 0;
    value_type * m_body = nullptr;
}; /* end class SimpleArray */

template <typename A, typename T>
SimpleArray<uint64_t> detail::SimpleArrayMixinSort<A, T>::argsort(void)
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(Formatter() << "SimpleArray::argsort(): currently only support 1D array"
                                             << " but the array is " << athis->ndim() << " dimension");
    }

    SimpleArray<uint64_t> ret(athis->shape());

    { // Return array initialization
        uint64_t cnt = 0;
        std::for_each(ret.begin(), ret.end(), [&cnt](uint64_t & v)
                      { v = cnt++; });
    }

    value_type const * buf = athis->body();
    auto cmp = [buf](uint64_t a, uint64_t b)
    {
        return buf[a] < buf[b];
    };
    std::sort(ret.begin(), ret.end(), cmp);
    return ret;
}

template <typename A, typename T>
template <typename I>
A detail::SimpleArrayMixinSort<A, T>::take_along_axis(SimpleArray<I> const & indices)
{
    static_assert(std::is_integral_v<I>, "I must be integral type");
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(Formatter() << "SimpleArray::take_along_axis(): currently only support 1D array"
                                             << " but the array is " << athis->ndim() << " dimension");
    }

    size_t max_idx = athis->shape()[0];
    I const * src = indices.begin();
    I const * const end = indices.end();
    while (src < end)
    {
        if (*src < 0 || *src > max_idx)
        {
            size_t offset = src - indices.begin();
            shape_type const & stride = indices.stride();
            Formatter err_msg;
            err_msg << "SimpleArray::take_along_axis(): indices[" << offset / stride[0];
            offset %= stride[0];
            for (size_t dim = 1; dim < stride.size(); ++dim)
            {
                err_msg << ", " << offset / stride[dim];
                offset %= stride[dim];
            }
            err_msg << "] is " << *src << ", which is out of range of the array size "
                    << max_idx;

            throw std::out_of_range(err_msg);
        }
        src++;
    }

    src = indices.begin();
    A ret(indices.shape());
    T * data = athis->begin();
    T * dst = ret.begin();
    while (src < end)
    {
        T * valp = data + static_cast<size_t>(*src);
        *dst = *valp;
        ++dst;
        ++src;
    }
    return ret;
}

template <typename T>
T const * detail::check_index_range(SimpleArray<T> const & indices, size_t max_idx)
{
    constexpr T DataTypeMax = std::numeric_limits<T>::max();
    constexpr T DataTypeMin = std::numeric_limits<T>::min();
    if (max_idx >= DataTypeMax && DataTypeMin == 0)
    {
        return nullptr;
    }

    return simd::check_between<T>(indices.begin(), indices.end(), 0, max_idx);
}

template <typename T, typename I>
void detail::indexed_copy(T * dest, T const * data, I const * index0, I const * const index1)
{
    T * dst = dest;
    I const * src = index0;
    while (src < index1)
    {
        T const * valp = data + static_cast<size_t>(*src);
        *dst = *valp;
        ++dst;
        ++src;
    }
}

template <typename A, typename T>
template <typename I>
A detail::SimpleArrayMixinSort<A, T>::take_along_axis_simd(SimpleArray<I> const & indices)
{
    static_assert(std::is_integral_v<I>, "I must be integral type");
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(Formatter() << "SimpleArray::take_along_axis(): currently only support 1D array"
                                             << " but the array is " << athis->ndim() << " dimension");
    }

    size_t max_idx = athis->shape()[0];

    I const * oor_ptr = check_index_range(indices, max_idx);
    if (oor_ptr != nullptr)
    {
        size_t offset = oor_ptr - indices.begin();
        shape_type const & stride = indices.stride();
        const size_t ndim = stride.size();
        Formatter err_msg;
        err_msg << "SimpleArray::take_along_axis_simd(): indices[" << offset / stride[0];
        offset %= stride[0];
        for (size_t dim = 1; dim < ndim; ++dim)
        {
            err_msg << ", " << offset / stride[dim];
            offset %= stride[dim];
        }
        err_msg << "] is " << *oor_ptr << ", which is out of range of the array size "
                << max_idx;

        throw std::out_of_range(err_msg);
    }

    I const * src = indices.begin();
    I const * const end = indices.end();
    A ret(indices.shape());
    T * data = athis->begin();
    T * dest = ret.begin();
    detail::indexed_copy(dest, data, src, end);
    return ret;
}

template <typename S>
using is_simple_array = std::is_same<
    std::remove_reference_t<S>,
    SimpleArray<typename std::remove_reference_t<S>::value_type>>;

template <typename S>
inline constexpr bool is_simple_array_v = is_simple_array<S>::value;

using SimpleArrayBool = SimpleArray<bool>;
using SimpleArrayInt8 = SimpleArray<int8_t>;
using SimpleArrayInt16 = SimpleArray<int16_t>;
using SimpleArrayInt32 = SimpleArray<int32_t>;
using SimpleArrayInt64 = SimpleArray<int64_t>;
using SimpleArrayUint8 = SimpleArray<uint8_t>;
using SimpleArrayUint16 = SimpleArray<uint16_t>;
using SimpleArrayUint32 = SimpleArray<uint32_t>;
using SimpleArrayUint64 = SimpleArray<uint64_t>;
using SimpleArrayFloat32 = SimpleArray<float>;
using SimpleArrayFloat64 = SimpleArray<double>;
using SimpleArrayComplex64 = SimpleArray<Complex<float>>;
using SimpleArrayComplex128 = SimpleArray<Complex<double>>;

class DataType
{
public:
    enum enum_type
    {
        Undefined,
        Bool,
        Int8,
        Int16,
        Int32,
        Int64,
        Uint8,
        Uint16,
        Uint32,
        Uint64,
        Float32,
        Float64,
        Complex64,
        Complex128
    }; /* end enum enum_type */

    DataType() = default;

    constexpr DataType(const enum_type datatype)
        : m_data_type(datatype)
    {
    }

    DataType(const std::string & data_type_string);

    enum_type type() const { return m_data_type; }

    constexpr operator enum_type() const { return m_data_type; } // Allow implicit switch and comparisons.
    explicit operator bool() const = delete; // Prevent usage: if(datatype)

    template <typename T>
    static DataType from();

private:
    enum_type m_data_type;

}; /* end class DataType */

class SimpleArrayPlex
{
public:
    using shape_type = detail::shape_type;

    SimpleArrayPlex() = default;

    explicit SimpleArrayPlex(const shape_type & shape, const std::string & data_type_string)
        : SimpleArrayPlex(shape, DataType(data_type_string))
    {
    }

    explicit SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, const std::string & data_type_string)
        : SimpleArrayPlex(shape, buffer, DataType(data_type_string))
    {
    }

    explicit SimpleArrayPlex(const shape_type & shape, const DataType data_type);
    explicit SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, const DataType data_type);

    template <typename T>
    SimpleArrayPlex(const SimpleArray<T> & array)
    {
        m_data_type = DataType::from<T>();
        m_has_instance_ownership = true;
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArray<T>(array));
    }

    SimpleArrayPlex(SimpleArrayPlex const & other);
    SimpleArrayPlex(SimpleArrayPlex && other) noexcept;
    SimpleArrayPlex & operator=(SimpleArrayPlex const & other);
    SimpleArrayPlex & operator=(SimpleArrayPlex && other) noexcept;

    ~SimpleArrayPlex();

    DataType data_type() const
    {
        return m_data_type;
    }

    /// Get the pointer to the const instance of SimpleArray<T>.
    const void * instance_ptr() const
    {
        return m_instance_ptr;
    }

    /// Get the pointer to the mutable instance of SimpleArray<T>.
    void * mutable_instance_ptr() const
    {
        return m_instance_ptr;
    }

    /// TODO: add all SimpleArray public methods

private:
    bool m_has_instance_ownership = false; /// ownership of the instance
    void * m_instance_ptr = nullptr; /// the pointer of the SimpleArray<T> instance
    DataType m_data_type = DataType::Undefined; /// the data type for array casting
}; /* end class SimpleArrayPlex */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
