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

struct CellTypeBase
{
    /* symbols for type id codes */
    static constexpr const uint8_t NONCELLTYPE   = 0; /* not a cell type */
    static constexpr const uint8_t POINT         = 1;
    static constexpr const uint8_t LINE          = 2;
    static constexpr const uint8_t QUADRILATERAL = 3;
    static constexpr const uint8_t TRIANGLE      = 4;
    static constexpr const uint8_t HEXAHEDRON    = 5;
    static constexpr const uint8_t TETRAHEDRON   = 6;
    static constexpr const uint8_t PRISM         = 7;
    static constexpr const uint8_t PYRAMID       = 8;
    /* number of all types; one larger than the last type id code */
    static constexpr const uint8_t NTYPE         = 8;

    //< Maximum number of nodes in a face.
    static constexpr const uint8_t FCNND_MAX = 4;
    //< Maximum number of cells in a face.
    static constexpr const uint8_t FCNCL_MAX = 2;
    //< Maximum number of nodes in a cell.
    static constexpr const uint8_t CLNND_MAX = 8;
    //< Maximum number of faces in a cell.
    static constexpr const uint8_t CLNFC_MAX = 6;
}; /* end struct CellTypeBase */

/**
 * Cell type for unstructured mesh.
 */
template < uint8_t ND > struct CellType
  : public CellTypeBase
  , public SpaceBase<ND>
{

    static constexpr const uint8_t NDIM = SpaceBase<ND>::NDIM;

    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */
    CellType(uint8_t id_in, uint8_t nnode_in, uint8_t nedge_in, uint8_t nsurface_in)
      : m_id(id_in), m_nnode(nnode_in), m_nedge(nedge_in), m_nsurface(nsurface_in) {}

    uint8_t id() const { return m_id; }
    uint8_t ndim() const { return NDIM; }
    uint8_t nnode() const { return m_nnode; }
    uint8_t nedge() const { return m_nedge; }
    uint8_t nsurface() const { return m_nsurface; }

    uint8_t nface() const
    {
        if      constexpr (2 == NDIM) { return nedge()   ; }
        else if constexpr (3 == NDIM) { return nsurface(); }
        else                          { return 0         ; }
    }

    const char * name() const
    {
        switch (id()) {
        case POINT         /* 1 */: return "point"         ; break;
        case LINE          /* 2 */: return "line"          ; break;
        case QUADRILATERAL /* 3 */: return "quadrilateral" ; break;
        case TRIANGLE      /* 4 */: return "triangle"      ; break;
        case HEXAHEDRON    /* 5 */: return "hexahedron"    ; break;
        case TETRAHEDRON   /* 6 */: return "tetrahedron"   ; break;
        case PRISM         /* 7 */: return "prism"         ; break;
        case PYRAMID       /* 8 */: return "pyramid"       ; break;
        case NONCELLTYPE   /* 0 */:
        default        /* other */: return "noncelltype"   ; break;
        }
    }

private:

    uint8_t m_id = NONCELLTYPE;
    uint8_t m_nnode = 0;
    uint8_t m_nedge = 0;
    uint8_t m_nsurface = 0;

}; /* end struct CellType */

#define MM_DECL_CELL_TYPE(NAME, TYPE, NDIM, NNODE, NEDGE, NSURFACE) \
struct NAME##CellType : public CellType<NDIM> { \
    NAME##CellType() : CellType<NDIM>(TYPE, NNODE, NEDGE, NSURFACE) {} \
}; \
static_assert(sizeof(NAME##CellType) == 4);
//                                 id, ndim, nnode, nedge, nsurface
MM_DECL_CELL_TYPE(Point        ,    1,    0,     1,     0,        0 ) // point/node/vertex
MM_DECL_CELL_TYPE(Line         ,    2,    1,     2,     0,        0 ) // line/edge
MM_DECL_CELL_TYPE(Quadrilateral,    3,    2,     4,     4,        0 )
MM_DECL_CELL_TYPE(Triangle     ,    4,    2,     3,     3,        0 )
MM_DECL_CELL_TYPE(Hexahedron   ,    5,    3,     8,    12,        6 ) // hexahedron/brick
MM_DECL_CELL_TYPE(Tetrahedron  ,    6,    3,     4,     6,        4 )
MM_DECL_CELL_TYPE(Prism        ,    7,    3,     6,     9,        5 )
MM_DECL_CELL_TYPE(Pyramid      ,    8,    3,     5,     8,        5 )
#undef MH_DECL_CELL_TYPE

template < typename D /* derived type */, uint8_t ND >
class StaticMeshBase
  : public SpaceBase<ND>
  , public std::enable_shared_from_this<D>
{

private:

    class ctor_passkey {};

public:

    using space_base = SpaceBase<ND>;
    using int_type = typename space_base::int_type;
    using uint_type = typename space_base::uint_type;
    using real_type = typename space_base::real_type;

    static constexpr const auto NDIM = space_base::NDIM;
    static constexpr const uint8_t FCMND = CellTypeBase::FCNND_MAX;
    static constexpr const uint8_t FCMCL = CellTypeBase::FCNCL_MAX;
    static constexpr const uint8_t CLMND = CellTypeBase::CLNND_MAX;
    static constexpr const uint8_t CLMFC = CellTypeBase::CLNFC_MAX;
    static constexpr const uint8_t FCNCL = 4;
    static constexpr const uint8_t FCREL = 4;
    static constexpr const uint8_t BFREL = 3;

    template < typename ... Args >
    static std::shared_ptr<D> construct(Args && ... args)
    {
        return std::make_shared<D>(std::forward<Args>(args) ..., ctor_passkey());
    }

    StaticMeshBase(uint_type nnode, ctor_passkey const &)
      : m_nnode(nnode), m_nface(0), m_ncell(0)
      , m_nbound(0), m_ngstnode(0), m_ngstface(0), m_ngstcell(0)
      , m_ndcrd(std::vector<size_t>{nnode, NDIM})
      , m_fccnd(std::vector<size_t>{0, NDIM})
      , m_fcnml(std::vector<size_t>{0, NDIM})
      , m_fcara(std::vector<size_t>{0})
      , m_clcnd(std::vector<size_t>{0, NDIM})
      , m_clvol(std::vector<size_t>{0})
      , m_fctpn(std::vector<size_t>{0})
      , m_cltpn(std::vector<size_t>{0})
      , m_clgrp(std::vector<size_t>{0})
      , m_fcnds(std::vector<size_t>{0, FCMND})
      , m_fccls(std::vector<size_t>{0, FCMCL})
      , m_clnds(std::vector<size_t>{0, CLMND})
      , m_clfcs(std::vector<size_t>{0, CLMFC})
    {}
    StaticMeshBase() = delete;
    StaticMeshBase(StaticMeshBase const & ) = delete;
    StaticMeshBase(StaticMeshBase       &&) = delete;
    StaticMeshBase & operator=(StaticMeshBase const & ) = delete;
    StaticMeshBase & operator=(StaticMeshBase       &&) = delete;
    ~StaticMeshBase() = default;

    uint_type nnode() const { return m_nnode; }
    uint_type nface() const { return m_nface; }
    uint_type ncell() const { return m_ncell; }
    uint_type nbound() const { return m_nbound; }
    uint_type ngstnode() const { return m_ngstnode; }
    uint_type ngstface() const { return m_ngstface; }
    uint_type ngstcell() const { return m_ngstcell; }

// Shape data.
private:

    uint_type m_nnode = 0; ///< Number of nodes (interior).
    uint_type m_nface = 0; ///< Number of faces (interior).
    uint_type m_ncell = 0; ///< Number of cells (interior).
    uint_type m_nbound = 0; ///< Number of boundary faces.
    uint_type m_ngstnode = 0; ///< Number of ghost nodes.
    uint_type m_ngstface = 0; ///< Number of ghost faces.
    uint_type m_ngstcell = 0; ///< Number of ghost cells.
    // other block information.
    bool m_use_incenter = false; ///< While true, m_clcnd uses in-center for simplices.

// Data arrays.
#define MM_DECL_StaticMesh_ARRAY(TYPE, NAME) \
public: \
    SimpleArray<TYPE> const & NAME() const { return m_ ## NAME; } \
    SimpleArray<TYPE>       & NAME()       { return m_ ## NAME; } \
private: \
    SimpleArray<TYPE> m_ ## NAME

    // geometry arrays.
    MM_DECL_StaticMesh_ARRAY(real_type, ndcrd);
    MM_DECL_StaticMesh_ARRAY(real_type, fccnd);
    MM_DECL_StaticMesh_ARRAY(real_type, fcnml);
    MM_DECL_StaticMesh_ARRAY(real_type, fcara);
    MM_DECL_StaticMesh_ARRAY(real_type, clcnd);
    MM_DECL_StaticMesh_ARRAY(real_type, clvol);
    // meta arrays.
    MM_DECL_StaticMesh_ARRAY(int_type, fctpn);
    MM_DECL_StaticMesh_ARRAY(int_type, cltpn);
    MM_DECL_StaticMesh_ARRAY(int_type, clgrp);
    // connectivity arrays.
    MM_DECL_StaticMesh_ARRAY(int_type, fcnds);
    MM_DECL_StaticMesh_ARRAY(int_type, fccls);
    MM_DECL_StaticMesh_ARRAY(int_type, clnds);
    MM_DECL_StaticMesh_ARRAY(int_type, clfcs);

#undef MM_DECL_StaticMesh_ARRAY

}; /* end class StaticMeshBase */

class StaticMesh2d
  : public StaticMeshBase<StaticMesh2d, 2>
{
    using StaticMeshBase::StaticMeshBase;
}; /* end class StaticGrid2d */

class StaticMesh3d
  : public StaticMeshBase<StaticMesh3d, 3>
{
    using StaticMeshBase::StaticMeshBase;
}; /* end class StaticGrid3d */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
