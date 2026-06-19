#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/BufferExpander.hpp>
#include <solvcon/buffer/SimpleArray.hpp>

#include <limits>
#include <stdexcept>

namespace solvcon
{

template <typename T>
class SimpleCollector
{

private:

    using array_internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename array_internal_types::value_type;
    using expander_type = BufferExpander;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    explicit SimpleCollector(size_t length = 0, size_t alignment = 0)
        : m_expander(BufferExpander::construct(length * ITEMSIZE, alignment))
    {
    }

    // Always (forcefully) clone the input array when it is a const reference
    explicit SimpleCollector(SimpleArray<T> const & arr, size_t alignment = 0)
        : m_expander(BufferExpander::construct(arr.buffer().clone(), /*clone*/ false, alignment))
    {
    }

    // Allow sharing the buffer when the input array is an lvalue reference
    explicit SimpleCollector(SimpleArray<T> & arr, bool clone, size_t alignment = 0)
        : m_expander(BufferExpander::construct(arr.buffer().shared_from_this(), clone, alignment))
    {
    }

    SimpleCollector(SimpleCollector const & other)
        : m_expander(other.m_expander->clone())
    {
    }

    SimpleCollector & operator=(SimpleCollector const & other)
    {
        if (this != &other)
        {
            m_expander = other.m_expander->clone();
        }
        return *this;
    }

    SimpleCollector(SimpleCollector && other) noexcept
        : m_expander(std::move(other.m_expander))
    {
    }

    SimpleCollector & operator=(SimpleCollector && other) noexcept
    {
        if (this != &other)
        {
            m_expander = std::move(other.m_expander);
        }
        return *this;
    }

    ~SimpleCollector() = default;

    size_t size() const { return expander().size() / ITEMSIZE; }
    size_t capacity() const { return expander().capacity() / ITEMSIZE; }
    size_t alignment() const { return expander().alignment(); }

    value_type const & operator[](size_t it) const noexcept { return data(it); }
    value_type & operator[](size_t it) noexcept { return data(it); }

    void reserve(size_t cap) { expander().reserve(cap * ITEMSIZE); }
    void expand(size_t length) { expander().expand(length * ITEMSIZE); }
    void clear() { expander().clear(); }

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

    SimpleArray<T> as_array()
    {
        return SimpleArray<T>(expander().as_concrete());
    }

    void push_back(value_type const & value)
    {
        size_t const it = size();
        push_size();
        (*this)[it] = value;
    }

    void push_back(value_type && value)
    {
        size_t const it = size();
        push_size();
        (*this)[it] = std::move(value);
    }

    /* Backdoor */
    value_type const & data(size_t it) const { return data()[it]; }
    value_type & data(size_t it) { return data()[it]; }
    value_type const * data() const { return expander().template data<value_type>(); }
    value_type * data() { return expander().template data<value_type>(); }

    expander_type const & expander() const { return *m_expander; }
    expander_type & expander() { return *m_expander; }

private:

    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            throw std::out_of_range(
                std::format("SimpleCollector: index {} is out of bounds with size {}",
                            it,
                            size()));
        }
    }

    /**
     * Push up the size of the underneath expander by one ITEMSIZE.
     */
    void push_size()
    {
        if (capacity() == 0)
        {
            reserve(1);
        }
        else if (size() == capacity())
        {
            reserve(size() * 2);
        }
        else
        {
            // do nothing
        }
        m_expander->push_size(ITEMSIZE);
    }

    std::shared_ptr<expander_type> m_expander;

}; /* end class SimpleCollector */

} /* end namespace solvcon */

/* vim: set et ts=4 sw=4: */
