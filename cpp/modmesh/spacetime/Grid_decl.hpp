#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>
#include <vector>

#include <modmesh/spacetime/base_spacetime.hpp>
#include <modmesh/spacetime/ElementBase_decl.hpp>

#include <modmesh/modmesh.hpp>

namespace spacetime
{

class Celm;
class Selm;

class Grid
    : public std::enable_shared_from_this<Grid>
{

public:

    // Remove the two aliases duplicated in ElementBase.
    using value_type = real_type;
    using array_type = modmesh::AscendantGrid1d::array_type;
    constexpr static size_t BOUND_COUNT = 2;
    static_assert(BOUND_COUNT >= 2, "BOUND_COUNT must be greater or equal to 2");

private:

    class ctor_passkey
    {
    };

public:

    template <class... Args>
    static std::shared_ptr<Grid> construct(Args &&... args)
    {
        return std::make_shared<Grid>(std::forward<Args>(args)..., ctor_passkey());
    }

    std::shared_ptr<Grid> clone() const
    {
        return std::make_shared<Grid>(*this);
    }

    Grid(real_type xmin, real_type xmax, size_t ncelm, ctor_passkey const &);

    // NOLINTNEXTLINE(hicpp-member-init,cppcoreguidelines-pro-type-member-init)
    Grid(array_type const & xloc, ctor_passkey const &) { init_from_array(xloc); }

    Grid() = delete;
    Grid(Grid const &) = default;
    Grid(Grid &&) = default;
    Grid & operator=(Grid const &) = default;
    Grid & operator=(Grid &&) = default;
    ~Grid() = default;

    real_type xmin() const { return m_xmin; }
    real_type xmax() const { return m_xmax; }
    size_t ncelm() const { return m_ncelm; }
    size_t nselm() const { return m_ncelm + 1; }

    size_t xsize() const { return m_agrid.size(); }

    array_type const & xcoord() const { return m_agrid.coord(); }
    array_type & xcoord() { return m_agrid.coord(); }

public:

    class CelmPK
    {
    private:
        CelmPK() = default;
        friend Celm;
    };

    /**
     * Get pointer to an coordinate value using conservation-element index.
     */
    real_type * xptr_celm(int_type ielm, bool odd_plane, CelmPK const &) { return xptr(xindex_celm(ielm, odd_plane)); }
    real_type const * xptr_celm(int_type ielm, bool odd_plane, CelmPK const &) const { return xptr(xindex_celm(ielm, odd_plane)); }

public:

    class SelmPK
    {
    private:
        SelmPK() = default;
        friend Selm;
    };

    /**
     * Get pointer to an coordinate value using conservation-element index.
     */
    real_type * xptr_selm(int_type ielm, bool odd_plane, SelmPK const &) { return xptr(xindex_selm(ielm, odd_plane)); }
    real_type const * xptr_selm(int_type ielm, bool odd_plane, SelmPK const &) const { return xptr(xindex_selm(ielm, odd_plane)); }

private:

    /**
     * Convert celm index to coordinate index.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    size_t xindex_celm(int_type ielm) const { return 1 + BOUND_COUNT + (ielm << 1); }
    size_t xindex_celm(int_type ielm, bool odd_plane) const { return xindex_celm(ielm) + (odd_plane ? 1 : 0); }

    /**
     * Convert selm index to coordinate index.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    size_t xindex_selm(int_type ielm) const { return BOUND_COUNT + (ielm << 1); }
    size_t xindex_selm(int_type ielm, bool odd_plane) const { return xindex_selm(ielm) + (odd_plane ? 1 : 0); }

    /**
     * Get pointer to an coordinate value using coordinate index.
     */
    real_type * xptr() { return m_agrid.data(); }
    real_type const * xptr() const { return m_agrid.data(); }
    real_type * xptr(size_t xindex) { return m_agrid.data() + xindex; }
    real_type const * xptr(size_t xindex) const { return m_agrid.data() + xindex; }

    void init_from_array(array_type const & xloc);

    real_type m_xmin;
    real_type m_xmax;
    size_t m_ncelm;

    modmesh::AscendantGrid1d m_agrid;

    template <class ET>
    friend class ElementBase;

}; /* end class Grid */

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
