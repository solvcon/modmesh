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
#include <modmesh/simd/neon.hpp>

#include <limits>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif /* defined(__aarch64__) */

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

template <typename A, typename T>
class SimpleArrayMixinCalculators
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

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
}; /* end class SimpleArrayMixinCalculators */

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
#if defined(__aarch64__)
    template <typename I>
    A take_along_axis_neon(SimpleArray<I> const & indices);
#endif /* defined(__aarch64__) */

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

#if defined(__aarch64__)

template <typename T, typename I>
void buffer_cpy(T * dest, T const * data, I const * begin, I const * const end);

template <typename T>
T const * check_index_range(SimpleArray<T> const & indices, size_t max_idx);

#endif /* defined(__aarch64__) */

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
            const size_t nbytes = ITEMSIZE *
                                  std::accumulate(shape.begin(),
                                                  shape.end(),
                                                  static_cast<size_t>(1),
                                                  std::multiplies<size_t>());
            if (nbytes != buffer->nbytes())
            {
                throw std::runtime_error(Formatter() << "SimpleArray: shape byte count " << nbytes
                                                     << " differs from buffer " << buffer->nbytes());
            }

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

    explicit operator bool() const noexcept { return bool(m_buffer) && bool(*m_buffer); }

    size_t nbytes() const noexcept { return m_buffer ? m_buffer->nbytes() : 0; }
    size_t size() const noexcept { return nbytes() / ITEMSIZE; }

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
        ssize_t dim = shape().size() - 1;
        while (dim >= 0)
        {
            sidx[dim] += 1;
            if (sidx[dim] == shape()[dim])
            {
                sidx[dim] = 0;
                dim -= 1;
                continue;
            }
            return true;
        }
        return false;
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
        if (it >= static_cast<ssize_t>(size() - m_nghost))
        {
            throw std::out_of_range(
                Formatter() << "SimpleArray: index " << it << " >= " << size() - m_nghost
                            << " (size: " << size() << " - nghost: " << m_nghost << ")");
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

#if defined(__aarch64__)
#define check_type_range(_type, max_val)                                \
    do {                                                                \
        constexpr _type __type_max = std::numeric_limits<_type>::max(); \
        constexpr _type __type_min = std::numeric_limits<_type>::min(); \
        if (max_val >= __type_max && __type_min == 0)                   \
        {                                                               \
            return nullptr;                                             \
        }                                                               \
    } while (0)

template <typename T>
T const * detail::check_index_range(SimpleArray<T> const & indices, size_t max_idx)
{
    check_type_range(T, max_idx);

    T const * src = indices.begin();
    T const * const end = indices.end();
    while (src < end)
    {
        T const idx = *src;
        if (idx < 0 || idx > max_idx)
        {
            return src;
        }
        ++src;
    }
    return nullptr;
}

#define DECL_MM_DECL_CHECK_IDX_RNG_NEON(_type) \
    template <>                                \
    _type const * detail::check_index_range<_type>(SimpleArray<_type> const & indices, size_t max_idx)

DECL_MM_DECL_CHECK_IDX_RNG_NEON(uint8_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(uint16_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(uint32_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(uint64_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(int8_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(int16_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(int32_t);
DECL_MM_DECL_CHECK_IDX_RNG_NEON(int64_t);

#undef DECL_MM_DECL_CHECK_IDX_RNG_NEON

template <typename T, typename I>
void detail::buffer_cpy(T * dest, T const * data, I const * begin, I const * const end)
{
    T * dst = dest;
    I const * src = begin;
    while (src < end)
    {
        T const * valp = data + static_cast<size_t>(*src);
        *dst = *valp;
        ++dst;
        ++src;
    }
}

template <typename A, typename T>
template <typename I>
A detail::SimpleArrayMixinSort<A, T>::take_along_axis_neon(SimpleArray<I> const & indices)
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
        err_msg << "SimpleArray::take_along_axis(): indices[" << offset / stride[0];
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
    detail::buffer_cpy(dest, data, src, end);
    return ret;
}
#endif /* defined(__aarch64__) */

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
    SimpleArrayPlex(SimpleArrayPlex && other);
    SimpleArrayPlex & operator=(SimpleArrayPlex const & other);
    SimpleArrayPlex & operator=(SimpleArrayPlex && other);

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
