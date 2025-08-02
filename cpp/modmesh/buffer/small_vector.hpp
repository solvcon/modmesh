#pragma once

/*
 * Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
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

#include <stdexcept>
#include <array>
#include <vector>
#include <algorithm>

// TODO: Solve circular include between <modmesh/toggle/toggle.hpp> and SimpleArray class.
// buffer/ (higher level) should depend on toggle/ (lower level).
#include <modmesh/toggle/RadixTree.hpp>

namespace modmesh
{

template <typename T, size_t N = 3>
class small_vector
{

public:

    using value_type = T;
    using iterator = T *;
    using const_iterator = T const *;

    explicit small_vector(size_t size)
        : m_size(static_cast<unsigned int>(size))
    {
        if (m_size > N)
        {
            m_capacity = m_size;
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            m_head = new T[m_capacity];
        }
        else
        {
            m_capacity = N;
            m_head = m_data.data();
        }
    }

    explicit small_vector(size_t size, T const & v)
        : small_vector(size)
    {
        std::fill(begin(), end(), v);
    }

    explicit small_vector(std::vector<T> const & vector)
        : small_vector(vector.size())
    {
        std::copy_n(vector.begin(), m_size, begin());
    }

    template <class InputIt>
    small_vector(InputIt first, InputIt last)
        : small_vector(last - first)
    {
        std::copy(first, last, begin());
    }

    small_vector(std::initializer_list<T> init)
        : small_vector(init.size())
    {
        std::copy_n(init.begin(), m_size, begin());
    }

    small_vector() { m_head = m_data.data(); }

    small_vector(small_vector const & other)
        : m_size(other.m_size)
    {
        if (other.m_head == other.m_data.data())
        {
            m_capacity = N;
            m_head = m_data.data();
        }
        else
        {
            m_capacity = m_size;
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            m_head = new T[m_capacity];
        }
        std::copy_n(other.m_head, m_size, m_head);
    }

    small_vector(small_vector && other) noexcept
        : m_size(other.m_size)
    {
        if (other.m_head == other.m_data.data())
        {
            m_capacity = N;
            std::copy_n(other.m_data.begin(), m_size, m_data.begin());
            m_head = m_data.data();
        }
        else
        {
            m_capacity = m_size;
            m_head = other.m_head;
            other.m_size = 0;
            other.m_capacity = N;
            other.m_head = other.m_data.data();
        }
    }

    small_vector & operator=(small_vector const & other)
    {
        if (this != &other)
        {
            if (other.m_head == other.m_data.data())
            {
                if (m_head != m_data.data())
                {
                    delete[] m_head;
                    m_head = m_data.data();
                }
                m_size = other.m_size;
                m_capacity = N;
            }
            else
            {
                if (m_capacity < other.m_size && m_head != m_data.data())
                {
                    delete[] m_head;
                    m_head = nullptr;
                }
                if (m_head == m_data.data() || m_head == nullptr)
                {
                    m_capacity = other.m_size;
                    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                    m_head = new T[m_capacity];
                }
                m_size = other.m_size;
            }
            std::copy_n(other.m_head, m_size, m_head);
        }
        return *this;
    }

    small_vector & operator=(small_vector && other) noexcept
    {
        if (this != &other)
        {
            if (other.m_head == other.m_data.data())
            {
                if (m_head != m_data.data())
                {
                    delete[] m_head;
                    m_head = m_data.data();
                }
                m_size = other.m_size;
                m_capacity = N;
                std::copy_n(other.m_data.begin(), m_size, m_data.begin());
            }
            else
            {
                m_size = other.m_size;
                m_capacity = other.m_capacity;
                m_head = other.m_head;
                other.m_size = 0;
                other.m_capacity = N;
                other.m_head = other.m_data.data();
            }
        }
        return *this;
    }

    small_vector & operator=(std::vector<T> const & other)
    {
        if (size() < other.size())
        {
            std::copy(other.begin(), other.begin() + size(), begin());
            for (size_t it = size(); it < other.size(); ++it)
            {
                push_back(other[it]);
            }
        }
        else
        {
            std::copy(other.begin(), other.end(), begin());
            m_size = static_cast<unsigned int>(other.size());
        }
        return *this;
    }

    ~small_vector()
    {
        if (m_head != m_data.data() && m_head != nullptr)
        {
            delete[] m_head;
            m_head = nullptr;
        }
    }

    size_t size() const noexcept { return m_size; }
    size_t capacity() const noexcept { return m_capacity; }
    bool empty() const noexcept { return 0 == m_size; }

    iterator begin() noexcept { return m_head; }
    iterator end() noexcept { return m_head + m_size; }
    const_iterator begin() const noexcept { return m_head; }
    const_iterator end() const noexcept { return m_head + m_size; }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    T const & operator[](size_t it) const { return m_head[it]; }
    T & operator[](size_t it) { return m_head[it]; }

    T const & at(size_t it) const
    {
        validate_range(it);
        return (*this)[it];
    }
    T & at(size_t it)
    {
        validate_range(it);
        return (*this)[it];
    }

    T const * data() const { return m_head; }
    T * data() { return m_head; }

    void clear() noexcept
    {
        if (m_head != m_data.data())
        {
            delete[] m_head;
            m_head = m_data.data();
        }
        m_size = 0;
        m_capacity = N;
    }

    void push_back(T const & value)
    {
        if (m_size == m_capacity)
        {
            m_capacity *= 2;
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            T * storage = new T[m_capacity];
            std::copy_n(m_head, m_size, storage);
            if (m_head != m_data.data())
            {
                delete[] m_head;
            }
            m_head = storage;
        }
        m_head[m_size++] = value;
    }

    size_t count(T const & value) const
    {
        size_t count = 0;
        for (const_iterator it = begin(); it != end(); ++it)
        {
            if (*it == value)
            {
                ++count;
            }
        }
        return count;
    }

    bool next_cartesian_product(small_vector const & bound)
    {
        int64_t dim = size() - 1;
        while (dim >= 0)
        {
            (*this)[dim] += 1;
            if ((*this)[dim] < bound[dim])
            {
                return true;
            }
            (*this)[dim] = 0;
            dim -= 1;
        }
        return false;
    }

    T select_kth(size_t k)
    {
        USE_CALLPROFILER_PROFILE_THIS_SCOPE("small_vector::select_kth()");
        iterator it = quick_select(begin(), end(), k);
        return *it;
    }

    iterator choose_pivot(iterator left, iterator right)
    {
        iterator first = left;
        iterator mid = left + (right - left) / 2;
        iterator last = right - 1;
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

    iterator quick_select(iterator first, iterator last, size_t k)
    {
        size_t len = last - first;
        if (k >= len)
        {
            throw std::out_of_range("quick_select: k out of range");
        }
        iterator left = first;
        iterator right = last;
        while (true)
        {
            iterator pivot_it = choose_pivot(left, right);
            std::iter_swap(pivot_it, right - 1);
            iterator store = left;
            for (iterator it = left; it < right - 1; ++it)
            {
                if (*it < *(right - 1))
                {
                    std::iter_swap(it, store);
                    ++store;
                }
            }
            std::iter_swap(store, right - 1);
            size_t pivot_rank = store - left;
            if (pivot_rank == k)
            {
                return store;
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

private:

    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            throw std::out_of_range("small_vector: index out of range");
        }
    }

    T * m_head = nullptr;
    unsigned int m_size = 0;
    unsigned int m_capacity = N;
    std::array<T, N> m_data;

}; /* end class small_vector */

template <typename T>
bool operator==(small_vector<T> const & lhs, small_vector<T> const & rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

static_assert(sizeof(small_vector<size_t>) == 40, "small_vector<size_t> should use 40 bytes");

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
