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

namespace modmesh
{

void BufferExpander::reserve(size_type cap)
{
    if (cap > capacity())
    {
        size_type const old_size = size();
        // Create new data holder and copy data.
        std::unique_ptr<int8_t, aligned_deleter> new_data_holder = allocate(cap, m_alignment);
        std::copy_n(m_begin, old_size, new_data_holder.get());
        // Process data holders.
        m_data_holder.swap(new_data_holder);
        if (m_concrete_buffer)
        {
            m_concrete_buffer.reset();
        }
        // Reset pointers.
        m_begin = m_data_holder.get();
        m_end = m_begin + old_size;
        m_end_cap = m_begin + cap;
    }
    else
    {
        return;
    }
}

std::shared_ptr<ConcreteBuffer> BufferExpander::copy_concrete(size_type cap) const
{
    size_type const old_size = size();
    size_type const csize = cap > old_size ? cap : old_size;
    auto buf = ConcreteBuffer::construct(csize);
    std::copy_n(m_begin, old_size, buf->data());
    return buf;
}

std::shared_ptr<ConcreteBuffer> const & BufferExpander::as_concrete(size_type cap)
{
    size_type const old_size = size();
    if (!m_concrete_buffer)
    {
        m_concrete_buffer = copy_concrete(cap);
        m_data_holder.reset();
    }
    m_begin = m_concrete_buffer->data();
    m_end = m_begin + old_size;
    m_end_cap = m_begin + m_concrete_buffer->size();
    return m_concrete_buffer;
}

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */