#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Structured grids.
 *
 * @ingroup group_mesh
 */

#include <solvcon/base.hpp>
#include <solvcon/toggle/toggle.hpp>
#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

/**
 * Base class template for a structured grid.
 *
 * @tparam ND Spatial dimension count.
 *
 * @ingroup group_mesh
 */
template <uint8_t ND>
class StaticGridBase : public SpaceBase<ND, int32_t, double>
{
}; /* end class StaticGridBase */

/**
 * One-dimensional grid whose coordinate ascends with the index.
 *
 * @ingroup group_mesh
 */
class AscendantGrid1d : public StaticGridBase<1>
{

public:

    using value_type = double;
    using array_type = SimpleArray<value_type>;

    explicit AscendantGrid1d(size_t ncoord)
        : m_coord(ncoord)
        , m_idmax(ncoord - 1)
    {
    }

    AscendantGrid1d() = default;
    AscendantGrid1d(AscendantGrid1d const &) = default;
    AscendantGrid1d(AscendantGrid1d &&) = default;
    AscendantGrid1d & operator=(AscendantGrid1d const &) = default;
    AscendantGrid1d & operator=(AscendantGrid1d &&) = default;
    ~AscendantGrid1d() = default;

    explicit operator bool() const { return static_cast<bool>(m_coord); }

    size_t ncoord() const { return m_idmax - m_idmin + 1; }

    size_t size() const { return m_coord.size(); }
    value_type const & operator[](size_t it) const { return m_coord[it]; }
    value_type & operator[](size_t it) { return m_coord[it]; }
    value_type const & at(size_t it) const { return m_coord.at(it); }
    value_type & at(size_t it) { return m_coord.at(it); }

    array_type const & coord() const { return m_coord; }
    array_type & coord() { return m_coord; }

    value_type const * data() const { return m_coord.data(); }
    value_type * data() { return m_coord.data(); }

private:

    array_type m_coord;
    size_t m_idmin = 0; // left internal boundary.
    size_t m_idmax = 0; // right internal boundary.

}; /* end class AscendantGrid1d */

/**
 * One-dimensional structured grid.
 *
 * @ingroup group_mesh
 */
class StaticGrid1d : public StaticGridBase<1>
{

public:

    using value_type = double;
    using array_type = SimpleArray<value_type>;

    // Constructs m_coord from nullptr, so it cannot use '= default'.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    StaticGrid1d()
        : m_coord(nullptr)
    {
    }

    explicit StaticGrid1d(serial_type nx)
        : m_nx(nx)
        , m_coord(nx)
    {
    }

    StaticGrid1d(StaticGrid1d const & other)
        : m_nx(other.m_nx)
        , m_coord(other.m_coord)
    {
    }

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
    {
    }

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
    array_type & coord() { return m_coord; }

    size_t size() const { return m_nx; }
    real_type operator[](size_t it) const noexcept { return m_coord[it]; }
    real_type & operator[](size_t it) noexcept { return m_coord[it]; }
    real_type at(size_t it) const { return m_coord.at(it); }
    real_type & at(size_t it) { return m_coord.at(it); }

    // Mutates the m_coord member, so it cannot be made static.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void fill(real_type val)
    {
        SOLVCON_PROFILE_SCOPE("StaticGrid1d::fill");
        std::fill(m_coord.begin(), m_coord.end(), val);
    }

private:

    serial_type m_nx = 0;
    array_type m_coord;

}; /* end class StaticGrid1d */

/**
 * Two-dimensional structured grid.
 *
 * @ingroup group_mesh
 */
class StaticGrid2d : public StaticGridBase<2>
{
}; /* end class StaticGrid2d */

/**
 * Three-dimensional structured grid.
 *
 * @ingroup group_mesh
 */
class StaticGrid3d : public StaticGridBase<3>
{
}; /* end class StaticGrid3d */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
