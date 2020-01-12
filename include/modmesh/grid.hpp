#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD-style license; see COPYING
 */

/**
 * Structured grid.
 */

#include "modmesh/base.hpp"
#include "modmesh/profile.hpp"
#include "modmesh/SimpleArray.hpp"

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

    StaticGrid1d() : m_nx(0), m_coord(nullptr) {}

    StaticGrid1d(serial_type nx)
      : m_nx(nx)
      , m_coord(allocate(nx))
    {}

    StaticGrid1d(StaticGrid1d const & other)
      : m_nx(other.m_nx)
      , m_coord(allocate(other.m_nx))
    {
        if (m_coord)
        {
            std::copy(other.m_coord.get(), m_coord.get(), m_coord.get()+m_nx);
        }
    }

    StaticGrid1d & operator=(StaticGrid1d const & other)
    {
        if (this != &other)
        {
            m_nx = other.m_nx;
            m_coord = allocate(m_nx);
            std::copy(other.m_coord.get(), m_coord.get(), m_coord.get()+m_nx);
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
    real_type * const coord() const { return m_coord.get(); }
    real_type *       coord()       { return m_coord.get(); }

    size_t size() const { return m_nx; }
    real_type   operator[] (size_t it) const noexcept { return m_coord[it]; }
    real_type & operator[] (size_t it)       noexcept { return m_coord[it]; }
    real_type   at (size_t it) const { ensure_range(it); return (*this)[it]; }
    real_type & at (size_t it)       { ensure_range(it); return (*this)[it]; }

    void fill(real_type val)
    {
        MODMESH_TIME("StaticGrid1d::fill");
        std::fill(m_coord.get(), m_coord.get()+m_nx, val);
    }

private:

    std::unique_ptr<real_type[]> allocate(serial_type nx)
    {
        if (nx)
        {
            return std::unique_ptr<real_type[]>(new real_type[nx]);
        }
        else
        {
            return std::unique_ptr<real_type[]>();
        }
    }

    void ensure_range(size_t it) const
    {
        if (it >= m_nx)
        {
            MODMESH_EXCEPT(StaticGrid1d, std::out_of_range, "index out of range");
        }
    }

    serial_type m_nx;
    std::unique_ptr<real_type[]> m_coord;

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
