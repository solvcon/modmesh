#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * This file is shared by all code in the view directory.
 */

#include <solvcon/python/python.hpp> // Must be the first include.
#include <solvcon/solvcon.hpp>

#include <QByteArray>

namespace solvcon
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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
