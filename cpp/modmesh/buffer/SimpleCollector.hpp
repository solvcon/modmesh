#pragma once

/*
 * Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/buffer/BufferExpander.hpp>
#include <modmesh/buffer/SimpleArray.hpp>

#include <limits>
#include <stdexcept>

namespace modmesh
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

    explicit SimpleCollector(size_t length)
        : m_expander(BufferExpander::construct(length * ITEMSIZE))
    {
    }

    // Always (forcefully) clone the input array when it is a const reference
    SimpleCollector(SimpleArray<T> const & arr)
        : m_expander(BufferExpander::construct(arr.buffer().clone(), /*clone*/ false))
    {
    }

    // Allow sharing the buffer when the input array is an lvalue reference
    SimpleCollector(SimpleArray<T> & arr, bool clone)
        : m_expander(BufferExpander::construct(arr.buffer().shared_from_this(), clone))
    {
    }

    SimpleCollector()
        : m_expander(BufferExpander::construct())
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

    SimpleCollector(SimpleCollector && other)
        : m_expander(std::move(other.m_expander))
    {
    }

    SimpleCollector & operator=(SimpleCollector && other)
    {
        if (this != &other)
        {
            m_expander = std::move(other.m_expander);
        }
        return *this;
    }

    size_t size() const { return expander().size() / ITEMSIZE; }
    size_t capacity() const { return expander().capacity() / ITEMSIZE; }

    value_type const & operator[](size_t it) const noexcept { return data(it); }
    value_type & operator[](size_t it) noexcept { return data(it); }

    void reserve(size_t cap) { expander().reserve(cap * ITEMSIZE); }
    void expand(size_t length) { expander().expand(length * ITEMSIZE); }

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
        (*this)[it] = value;
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
            throw std::out_of_range(Formatter() << "SimpleCollector: index " << it << " is out of bounds with size " << size());
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

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
