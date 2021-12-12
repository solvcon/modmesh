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

#include <modmesh/ConcreteBuffer.hpp>

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
template < typename T >
class SimpleArray
{

public:

    using value_type = T;
    using shape_type = small_vector<size_t>;
    using sshape_type = small_vector<ssize_t>;
    using buffer_type = ConcreteBuffer;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    static constexpr size_t itemsize() { return ITEMSIZE; }

    explicit SimpleArray(size_t length)
      : m_buffer(buffer_type::construct(length * ITEMSIZE))
      , m_shape{length}
      , m_stride{1}
      , m_body(m_buffer->data<T>())
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
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE);
            m_body = m_buffer->data<T>();
        }
    }

    explicit SimpleArray(std::vector<size_t> const & shape)
      : m_shape(shape), m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty()) {
            m_buffer = buffer_type::construct(m_shape[0] * m_stride[0] * ITEMSIZE);
            m_body = m_buffer->data<T>();
        }
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
            m_body = m_buffer->data<T>();
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
      , m_nghost(other.m_nghost)
      , m_body(calc_body(m_buffer->data<T>(), m_stride, other.m_nghost))
    {}

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
            m_shape = other.m_shape;
            m_stride = other.m_stride;
            m_nghost = other.m_nghost;
            m_body = calc_body(m_buffer->data<T>(), m_stride, other.m_nghost);
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

    value_type const & at(ssize_t it) const { validate_range(it); it += m_nghost; return data(it); }
    value_type       & at(ssize_t it)       { validate_range(it); it += m_nghost; return data(it); }

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

    value_type const & at(std::vector<ssize_t> const & idx) const { return at(sshape_type(idx)); }
    value_type       & at(std::vector<ssize_t> const & idx)       { return at(sshape_type(idx)); }

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

    size_t ndim() const noexcept { return m_shape.size(); }
    shape_type const & shape() const { return m_shape; }
    size_t   shape(size_t it) const noexcept { return m_shape[it]; }
    size_t & shape(size_t it)       noexcept { return m_shape[it]; }
    shape_type const & stride() const { return m_stride; }
    size_t   stride(size_t it) const noexcept { return m_stride[it]; }
    size_t & stride(size_t it)       noexcept { return m_stride[it]; }

    size_t nghost() const { return m_nghost; }
    size_t nbody() const { return m_shape.empty() ? 0 : m_shape[0] - m_nghost; }
    bool has_ghost() const { return m_nghost != 0; }
    void set_nghost(size_t nghost)
    {
        if (0 != nghost)
        {
            if (0 == ndim())
            {
                std::ostringstream ms;
                ms << "SimpleArray: cannot set nghost " << nghost << " > 0 to an empty array";
                throw std::out_of_range(ms.str());
            }
            if (nghost > shape(0))
            {
                std::ostringstream ms;
                ms << "SimpleArray: cannt set nghost " << nghost << " > shape(0) " << shape(0);
                throw std::out_of_range(ms.str());
            }
        }
        m_nghost = nghost;
        if (bool(*this))
        {
            m_body = calc_body(m_buffer->data<T>(), m_stride, m_nghost);
        }
    }

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

    void swap(SimpleArray<T> && other)
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

    template < typename ... Args >
    value_type const & operator()(Args ... args) const { return m_body[buffer_offset(m_stride, args...)]; }

    template < typename ... Args >
    value_type       & operator()(Args ... args)       { return m_body[buffer_offset(m_stride, args...)]; }

    /* Backdoor */
    value_type const & data(size_t it) const { return data()[it]; }
    value_type       & data(size_t it)       { return data()[it]; }
    value_type const * data() const { return buffer().template data<value_type>(); }
    value_type       * data()       { return buffer().template data<value_type>(); }

    buffer_type const & buffer() const { return *m_buffer; }
    buffer_type       & buffer()       { return *m_buffer; }

    value_type const * body() const { return m_body; }
    value_type       * body()       { return m_body; }

private:

    void validate_range(ssize_t it) const
    {
        if (m_nghost != 0 && ndim() != 1)
        {
            std::ostringstream ms;
            ms << "SimpleArray::validate_range(): cannot handle "
               << ndim() << "-dimensional (more than 1) array with non-zero nghost: " << m_nghost;
            throw std::out_of_range(ms.str());
        }
        if (it < -static_cast<ssize_t>(m_nghost))
        {
            std::ostringstream ms;
            ms << "SimpleArray: index " << it << " < -nghost: " << -static_cast<ssize_t>(m_nghost);
            throw std::out_of_range(ms.str());
        }
        if (it >= static_cast<ssize_t>(size() - m_nghost))
        {
            std::ostringstream ms;
            ms << "SimpleArray: index " << it << " >= " << size() - m_nghost
               << " (size: " << size() << " - nghost: " << m_nghost << ")";
            throw std::out_of_range(ms.str());
        }
    }

    void validate_shape(small_vector<ssize_t> const & idx) const
    {
        auto index2string = [&idx]()
        {
            std::ostringstream ms;
            ms << "[";
            for (size_t it = 0 ; it < idx.size() ; ++it)
            {
                ms << idx[it];
                if (it != idx.size()-1) { ms << ", "; }
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
            std::ostringstream ms;
            ms << "SimpleArray: dimension of input indices " << index2string() << " != array dimension " << m_shape.size();
            throw std::out_of_range(ms.str());
        }

        // Test the first dimension.
        if (idx[0] < -static_cast<ssize_t>(m_nghost))
        {
            std::ostringstream ms;
            ms << "SimpleArray: dim 0 in " << index2string() << " < -nghost: " << -static_cast<ssize_t>(m_nghost);
            throw std::out_of_range(ms.str());
        }
        if (idx[0] >= static_cast<ssize_t>(nbody()))
        {
            std::ostringstream ms;
            ms << "SimpleArray: dim 0 in " << index2string() << " >= nbody: " << nbody()
                << " (shape[0]: " << m_shape[0] << " - nghost: " << nghost() << ")";
            throw std::out_of_range(ms.str());
        }

        // Test the rest of the dimensions.
        for (size_t it = 1 ; it < m_shape.size() ; ++it)
        {
            if (idx[it] < 0)
            {
                std::ostringstream ms;
                ms << "SimpleArray: dim " << it << " in " << index2string() << " < 0";
                throw std::out_of_range(ms.str());
            }
            if (idx[it] >= static_cast<ssize_t>(m_shape[it]))
            {
                std::ostringstream ms;
                ms << "SimpleArray: dim " << it << " in " << index2string()
                   << " >= shape[" << it << "]: " << m_shape[it];
                throw std::out_of_range(ms.str());
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

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
