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
class BufferBase
{
public:

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    virtual ~BufferBase() = default;

    virtual explicit operator bool() const noexcept = 0;

    virtual size_type size() const noexcept = 0;
    virtual size_t nbytes() const noexcept = 0;
    virtual size_type capacity() const noexcept = 0;
    virtual void reserve(size_type cap) = 0;

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

    virtual iterator begin() noexcept = 0;
    virtual iterator end() noexcept = 0;
    virtual const_iterator begin() const noexcept = 0;
    virtual const_iterator end() const noexcept = 0;
    virtual const_iterator cbegin() const noexcept = 0;
    virtual const_iterator cend() const noexcept = 0;

    virtual int8_t data(size_type it) const = 0;
    virtual int8_t & data(size_type it) = 0;
    virtual int8_t const * data() const noexcept = 0;
    virtual int8_t * data() noexcept = 0;

protected:
    void validate_range(size_t it) const
    {
        if (it >= size())
        {
            throw std::out_of_range(Formatter() << name() << ": index " << it << " is out of bounds with size " << size());
        }
    }

    virtual constexpr std::string const & name() const { return "BufferBase"; }
};
} // namespace modmesh
