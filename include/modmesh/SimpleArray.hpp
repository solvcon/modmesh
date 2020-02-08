#pragma once

/*
 * Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
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


/**
 * Simple array type for contiguous memory storage. The copy semantics performs
 * data copy. The move semantics invalidates the existing memory buffer.
 */
template < typename T >
class SimpleArray
{

public:

    using value_type = T;
    using shape_type = small_vector<size_t>;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    static constexpr size_t itemsize() { return ITEMSIZE; }

    explicit SimpleArray(size_t length)
      : m_buffer(ConcreteBuffer::construct(length * ITEMSIZE))
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
        if (!m_shape.empty()) { m_buffer = ConcreteBuffer::construct(m_shape[0] * m_stride[0] * ITEMSIZE); }
    }

    explicit SimpleArray(std::shared_ptr<ConcreteBuffer> const & buffer)
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

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit SimpleArray
    (
        small_vector<size_t> const & shape
      , std::shared_ptr<ConcreteBuffer> const & buffer
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
                std::runtime_error("SimpleArray: input buffer size differs from shape");
            }
        }
    }

    explicit SimpleArray(std::vector<size_t> const & shape)
      : m_shape(shape), m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty()) { m_buffer = ConcreteBuffer::construct(m_shape[0] * m_stride[0] * ITEMSIZE); }
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

    ConcreteBuffer const & buffer() const { return *m_buffer; }
    ConcreteBuffer       & buffer()       { return *m_buffer; }

private:

    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            std::ostringstream msgstream;
            msgstream << "SimpleArray: index " << it << " is out of bounds with size " << size();
        }
    }

    std::shared_ptr<ConcreteBuffer> m_buffer;
    shape_type m_shape;
    shape_type m_stride;

}; /* end class SimpleArray */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
