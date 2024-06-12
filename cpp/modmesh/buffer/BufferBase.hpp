#pragma once
/*
 * Copyright (c) 2024, An-Chi Liu <phy.tiger@gmail.com>
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

#include <modmesh/base.hpp>

namespace modmesh
{

/// Base class for buffer-like objects.
class BufferBase : public std::enable_shared_from_this<BufferBase>
{
public:

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    virtual ~BufferBase() = default;

    virtual explicit operator bool() const { return bool(m_begin); }
    size_type size() const
    {
        return static_cast<size_type>(this->m_end - this->m_begin);
    }
    size_t nbytes() const { return size() * sizeof(int8_t); }

    int8_t operator[](size_t it) const { return data(it); }
    int8_t & operator[](size_t it) { return data(it); }

    int8_t at(size_t it) const
    {
        validate_range(it);
        return data(it);
    }

    int8_t & at(size_t it)
    {
        validate_range(it);
        return data(it);
    }

    using iterator = int8_t *;
    using const_iterator = int8_t const *;

    iterator begin() noexcept { return m_begin; }
    iterator end() noexcept { return m_end; }
    const_iterator begin() const noexcept { return m_begin; }
    const_iterator end() const noexcept { return m_end; }
    const_iterator cbegin() const noexcept { return m_begin; }
    const_iterator cend() const noexcept { return m_end; }

    /* Backdoor */
    int8_t data(size_type it) const { return data()[it]; }
    int8_t & data(size_type it) { return data()[it]; }
    int8_t const * data() const noexcept { return data<int8_t>(); }
    int8_t * data() noexcept { return data<int8_t>(); }

    // clang-format off
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template <typename T> T const * data() const noexcept { return reinterpret_cast<T *>(m_begin); }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template <typename T> T * data() noexcept { return reinterpret_cast<T *>(m_begin); }
    // clang-format on

protected:
    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            throw std::out_of_range(Formatter() << name() << ": index " << it << " is out of bounds with size " << size());
        }
    }

    // TODO: make this constexpr virtual once C++20 is available
    virtual const char * name() const { return "BufferBase"; }

    int8_t * m_begin = nullptr;
    int8_t * m_end = nullptr;
}; /* end class BufferBase */
} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */