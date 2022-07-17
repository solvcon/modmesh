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

// Shared by all code.

// TODO: Most of the code in view is highly coupled with Python; move the
// Python-related parts to modmesh/python/.

#include <modmesh/python/python.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

#include <QByteArray>

namespace modmesh
{

/**
 * @brief Return a SimpleArray from the buffer of the input QByteArray.
 *
 * @param view If true, the returned SimpleArray reuses (shares) the memory
 * buffer of the input array, and gives up (does not own) the memory buffer.
 */
template <typename T>
SimpleArray<T> makeSimpleArray(QByteArray & qbarr, small_vector<size_t> shape, bool view = false)
{
    size_t nbytes = sizeof(T);
    for (size_t v : shape)
    {
        nbytes *= v;
    }
    if (0 == nbytes)
    {
        return SimpleArray<T>(0);
    }
    if (qbarr.size() < 0 || nbytes != static_cast<size_t>(qbarr.size()))
    {
        throw std::out_of_range("QByteArray size disagrees with the requested shape");
    }

    std::shared_ptr<ConcreteBuffer> cbuf;
    if (view)
    {
        cbuf = ConcreteBuffer::construct(nbytes, qbarr.data(), std::make_unique<detail::ConcreteBufferNoRemove>());
    }
    else
    {
        cbuf = ConcreteBuffer::construct(nbytes);
        int8_t * ptr = reinterpret_cast<int8_t *>(qbarr.data());
        std::copy_n(ptr, nbytes, cbuf->begin());
    }

    return SimpleArray<T>(small_vector<size_t>(shape), cbuf);
}

template <typename T>
QByteArray makeQByteArray(SimpleArray<T> const & sarr)
{
    QByteArray barray;
    barray.resize(sarr.nbytes());
    std::copy_n(sarr.data(), sarr.size(), reinterpret_cast<T *>(barray.data()));
    return barray;
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
