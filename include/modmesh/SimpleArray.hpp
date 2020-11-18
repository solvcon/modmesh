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

#include "modmesh/ConcreteBuffer.hpp"

#include <stdexcept>

namespace modmesh
{

namespace detail
{

template < size_t D, typename S >
size_t buffer_offset_impl(S const &)
{
    return 0;
}

template < size_t D, typename S, typename Arg, typename ... Args >
size_t buffer_offset_impl(S const & strides, Arg arg, Args ... args)
{
    return arg * strides[D] + buffer_offset_impl<D+1>(strides, args...);
}

} /* end namespace detail */

template < typename S, typename ... Args >
size_t buffer_offset(S const & strides, Args ... args)
{
    return detail::buffer_offset_impl<0>(strides, args...);
}

inline size_t buffer_offset(small_vector<size_t> const & stride, small_vector<size_t> const & idx)
{
    if (stride.size() != idx.size())
    {
        std::ostringstream ms;
        ms << "stride size " << stride.size() << " != " << "index size " << idx.size();
        throw std::out_of_range(ms.str());
    }
    size_t offset = 0;
    for (size_t it = 0 ; it < stride.size() ; ++it)
    {
        offset += stride[it] * idx[it];
    }
    return offset;
}


/**
 * Simple array type for contiguous memory storage. Size does not change. The
 * copy semantics performs data copy. The move semantics invalidates the
 * existing memory buffer.
 */
template < typename T, typename D = ConcreteBufferDefaultDelete >
class SimpleArray
{

public:

    using value_type = T;
    using shape_type = small_vector<size_t>;
    using buffer_type = ConcreteBuffer<D>;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    static constexpr size_t itemsize() { return ITEMSIZE; }

    explicit SimpleArray(size_t length)
      : m_buffer(buffer_type::construct(length * ITEMSIZE))
      , m_shape{length}
      , m_stride{1}
    {}

    template< class InputIt > SimpleArray(InputIt first, InputIt last)
      : SimpleArray(last-first)
    {
        std::copy(first, last, data());
    }

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit SimpleArray(small_vector<size_t> const & shape)
      : m_shape(shape), m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty()) { m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE); }
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
        }
        else
        {
            throw std::runtime_error("SimpleArray: buffer cannot be null");
        }
    }

    explicit SimpleArray
    (
        small_vector<size_t> const & shape
      , std::shared_ptr<buffer_type> const & buffer
    )
      : SimpleArray(buffer)
    {
        if (buffer)
        {
            m_shape = shape;
            m_stride = calc_stride(m_shape);
            const size_t nbytes = m_shape[0] * m_stride[0] * ITEMSIZE;
            if (nbytes != buffer->nbytes())
            {
                std::ostringstream ms;
                ms << "SimpleArray: shape byte count " << nbytes << " differs from buffer " << buffer->nbytes();
                throw std::runtime_error(ms.str());
            }
        }
    }

    explicit SimpleArray(std::vector<size_t> const & shape)
      : m_shape(shape), m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty()) { m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE); }
    }

    static shape_type calc_stride(shape_type const & shape)
    {
        shape_type stride(shape.size());
        if (!shape.empty())
        {
            stride[shape.size()-1] = 1;
            for (size_t it=shape.size()-1; it>0; --it)
            {
                stride[it-1] = stride[it] * shape[it];
            }
        }
        return stride;
    }

    SimpleArray(std::initializer_list<T> init)
      : SimpleArray(init.size())
    {
        std::copy_n(init.begin(), init.size(), data());
    }

    SimpleArray() = default;

    SimpleArray(SimpleArray const & other)
      : m_buffer(other.m_buffer->clone())
      , m_shape(other.m_shape)
      , m_stride(other.m_stride)
    {}

    SimpleArray(SimpleArray && other) noexcept
      : m_buffer(std::move(other.m_buffer))
      , m_shape(std::move(other.m_shape))
      , m_stride(std::move(other.m_stride))
    {}

    SimpleArray & operator=(SimpleArray const & other)
    {
        if (this != &other)
        {
            *m_buffer = *(other.m_buffer); // Size is checked inside.
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
        }
        return *this;
    }

    ~SimpleArray() = default;

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
    value_type       & operator[](size_t it)       noexcept { return data(it); }

    value_type const & at(size_t it) const { validate_range(it); return data(it); }
    value_type       & at(size_t it)       { validate_range(it); return data(it); }

    value_type const & at(std::vector<size_t> const & idx) const { return at(shape_type(idx)); }
    value_type       & at(std::vector<size_t> const & idx)       { return at(shape_type(idx)); }

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

    size_t ndim() const noexcept { return m_shape.size(); }
    shape_type const & shape() const { return m_shape; }
    size_t   shape(size_t it) const noexcept { return m_shape[it]; }
    size_t & shape(size_t it)       noexcept { return m_shape[it]; }
    shape_type const & stride() const { return m_stride; }
    size_t   stride(size_t it) const noexcept { return m_stride[it]; }
    size_t & stride(size_t it)       noexcept { return m_stride[it]; }

    template < typename U >
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

    template < typename ... Args >
    value_type const & operator()(Args ... args) const { return data(buffer_offset(m_stride, args...)); }

    template < typename ... Args >
    value_type       & operator()(Args ... args)       { return data(buffer_offset(m_stride, args...)); }

    /* Backdoor */
    value_type const & data(size_t it) const { return data()[it]; }
    value_type       & data(size_t it)       { return data()[it]; }
    value_type const * data() const { return buffer().template data<value_type>(); }
    value_type       * data()       { return buffer().template data<value_type>(); }

    buffer_type const & buffer() const { return *m_buffer; }
    buffer_type       & buffer()       { return *m_buffer; }

private:

    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            std::ostringstream ms;
            ms << "SimpleArray: index " << it << " is out of bounds with size " << size();
            throw std::out_of_range(ms.str());
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

}; /* end class SimpleArray */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
