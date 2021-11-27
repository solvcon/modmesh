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

/**
 * Structured grid.
 */

#include <modmesh/base.hpp>
#include <modmesh/profile.hpp>
#include <modmesh/SimpleArray.hpp>

namespace modmesh
{

/**
 * Base class template for structured grid.
 */
template <size_t ND>
class StaticGridBase
  : public SpaceBase<ND>
{
}; /* end class StaticGridBase */

/**
 * 1D grid whose coordnate ascends with index.
 */
class AscendantGrid1d
  : public StaticGridBase<1>
{

public:

    using value_type = double;
    using array_type = SimpleArray<value_type>;

    explicit AscendantGrid1d(size_t ncoord)
      : m_coord(ncoord)
      , m_idmax(ncoord-1)
    {}

    AscendantGrid1d() = default;
    AscendantGrid1d(AscendantGrid1d const & ) = default;
    AscendantGrid1d(AscendantGrid1d       &&) = default;
    AscendantGrid1d & operator=(AscendantGrid1d const & ) = default;
    AscendantGrid1d & operator=(AscendantGrid1d       &&) = default;
    ~AscendantGrid1d() = default;

    explicit operator bool () const { return bool(m_coord); }

    size_t ncoord() const { return m_idmax - m_idmin + 1; }

    size_t size() const { return m_coord.size(); }
    value_type const & operator[] (size_t it) const { return m_coord[it]; }
    value_type       & operator[] (size_t it)       { return m_coord[it]; }
    value_type const & at(size_t it) const { return m_coord.at(it); }
    value_type       & at(size_t it)       { return m_coord.at(it); }

    array_type const & coord() const { return m_coord; }
    array_type       & coord()       { return m_coord; }

    value_type const * data() const { return m_coord.data(); }
    value_type       * data()       { return m_coord.data(); }

private:

    array_type m_coord;
    size_t m_idmin = 0; // left internal boundary.
    size_t m_idmax = 0; // right internal boundary.

}; /* end class AscendantGrid1d */

/**
 * 1D grid.
 */
class StaticGrid1d
  : public StaticGridBase<1>
{

public:

    using value_type = double;
    using array_type = SimpleArray<value_type>;

    StaticGrid1d() : m_coord(nullptr) {}

    explicit StaticGrid1d(serial_type nx)
      : m_nx(nx)
      , m_coord(nx)
    {}

    StaticGrid1d(StaticGrid1d const & other)
      : m_nx(other.m_nx)
      , m_coord(other.m_coord)
    {}

    StaticGrid1d & operator=(StaticGrid1d const & other)
    {
        if (this != &other)
        {
            m_nx = other.m_nx;
            m_coord = other.m_coord;
        }
        return *this;
    }

    StaticGrid1d(StaticGrid1d && other) noexcept
      : m_nx(other.m_nx)
      , m_coord(std::move(other.m_coord))
    {}

    StaticGrid1d & operator=(StaticGrid1d && other) noexcept
    {
        if (this != &other)
        {
            m_nx = other.m_nx;
            m_coord = std::move(other.m_coord);
        }
        return *this;
    }

    ~StaticGrid1d() = default;

    size_t nx() const { return m_nx; }
    array_type const & coord() const { return m_coord; }
    array_type       & coord()       { return m_coord; }

    size_t size() const { return m_nx; }
    real_type   operator[] (size_t it) const noexcept { return m_coord[it]; }
    real_type & operator[] (size_t it)       noexcept { return m_coord[it]; }
    real_type   at (size_t it) const { return m_coord.at(it); }
    real_type & at (size_t it)       { return m_coord.at(it); }

    void fill(real_type val)
    {
        MODMESH_TIME("StaticGrid1d::fill");
        std::fill(m_coord.begin(), m_coord.end(), val);
    }

private:

    serial_type m_nx = 0;
    array_type m_coord;

}; /* end class StaticGrid1d */

class StaticGrid2d
  : public StaticGridBase<2>
{
}; /* end class StaticGrid2d */

class StaticGrid3d
  : public StaticGridBase<3>
{
}; /* end class StaticGrid3d */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
