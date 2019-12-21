#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <algorithm>
#include <memory>

#define MODMESH_EXCEPT(CLS, EXC, MSG) throw EXC(#CLS ": " MSG);

namespace modmesh
{

/**
 * Spatial table basic information.  Any table-based data store for spatial
 * data should inherit this class template.
 */
template <size_t ND>
class SpaceBase
{

public:

    static constexpr const size_t NDIM = ND;

    using serial_type = uint32_t;
    using real_type = double;

}; /* end class SpaceBase */

/**
 * Base class template for structured grid.
 */
template <size_t ND>
class GridBase
  : public SpaceBase<ND>
{
}; /* end class GridBase */

/**
 * 1D grid.
 */
class Grid1d
  : public GridBase<1>
{

public:

    Grid1d() : m_nx(0), m_coord(nullptr) {}

    Grid1d(serial_type nx)
      : m_nx(nx)
      , m_coord(allocate(nx))
    {}

    Grid1d(Grid1d const & other)
      : m_nx(other.m_nx)
      , m_coord(allocate(other.m_nx))
    {
        if (m_coord)
        {
            std::copy(other.m_coord.get(), m_coord.get(), m_coord.get()+m_nx);
        }
    }

    Grid1d & operator=(Grid1d const & other)
    {
        if (this != &other)
        {
            m_nx = other.m_nx;
            m_coord = allocate(m_nx);
            std::copy(other.m_coord.get(), m_coord.get(), m_coord.get()+m_nx);
        }
        return *this;
    }

    Grid1d(Grid1d && other) noexcept
      : m_nx(other.m_nx)
      , m_coord(std::move(other.m_coord))
    {}

    Grid1d & operator=(Grid1d && other) noexcept
    {
        if (this != &other)
        {
            m_nx = other.m_nx;
            m_coord = std::move(other.m_coord);
        }
        return *this;
    }

    ~Grid1d() = default;

    size_t nx() const { return m_nx; }
    real_type * const coord() const { return m_coord.get(); }
    real_type *       coord()       { return m_coord.get(); }

    size_t size() const { return m_nx; }
    real_type   operator[] (size_t it) const noexcept { return m_coord[it]; }
    real_type & operator[] (size_t it)       noexcept { return m_coord[it]; }
    real_type   at (size_t it) const { ensure_range(it); return (*this)[it]; }
    real_type & at (size_t it)       { ensure_range(it); return (*this)[it]; }

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
            MODMESH_EXCEPT(Grid1d, std::out_of_range, "index out of range");
        }
    }

    serial_type m_nx;
    std::unique_ptr<real_type[]> m_coord;

}; /* end class Grid1d */

class Grid2d
  : public GridBase<2>
{
}; /* end class Grid2d */

class Grid3d
  : public GridBase<3>
{
}; /* end class Grid3d */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
