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

#include <modmesh/buffer/ConcreteBuffer.hpp>

namespace modmesh
{

std::shared_ptr<ConcreteBuffer> ConcreteBuffer::clone() const
{
    std::shared_ptr<ConcreteBuffer> ret = construct(nbytes());
    std::copy_n(data(), size(), (*ret).data());
    return ret;
}

ConcreteBuffer::ConcreteBuffer(size_t nbytes, const ctor_passkey &)
    : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
    , m_nbytes(nbytes)
    , m_data(allocate(nbytes))
{
    m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
    m_end = m_begin + m_nbytes;
}

// NOLINTNEXTLINE(readability-non-const-parameter)
ConcreteBuffer::ConcreteBuffer(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, const ctor_passkey &)
    : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
    , m_nbytes(nbytes)
    , m_data(data, data_deleter_type(std::move(remover)))
{
    m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
    m_end = m_begin + m_nbytes;
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#endif
// Avoid enabled_shared_from_this copy constructor
// NOLINTNEXTLINE(bugprone-copy-constructor-init)
ConcreteBuffer::ConcreteBuffer(ConcreteBuffer const & other)
    : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
    , m_nbytes(other.m_nbytes)
    , m_data(allocate(other.m_nbytes))
{
    m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
    m_end = m_begin + m_nbytes;
    if (size() != other.size())
    {
        throw std::out_of_range("Buffer size mismatch");
    }
    std::copy_n(other.data(), size(), data());
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
ConcreteBuffer & ConcreteBuffer::operator=(ConcreteBuffer const & other)
{
    if (this != &other)
    {
        if (size() != other.size())
        {
            throw std::out_of_range("Buffer size mismatch");
        }
        std::copy_n(other.data(), size(), data());
    }
    return *this;
}

ConcreteBuffer::unique_ptr_type ConcreteBuffer::allocate(size_t nbytes)
{
    unique_ptr_type ret(nullptr, data_deleter_type());
    if (0 != nbytes)
    {
        ret = unique_ptr_type(new int8_t[nbytes], data_deleter_type());
    }
    return ret;
}

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
