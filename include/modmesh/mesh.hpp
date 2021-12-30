#pragma once

/*
 * Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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
 * Unstructured mesh.
 */

#include <modmesh/base.hpp>
#include <modmesh/profile.hpp>
#include <modmesh/SimpleArray.hpp>

namespace modmesh
{
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

    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */
    StaticMeshBase(uint_type nnode, uint_type nface, uint_type ncell, uint_type nbound, ctor_passkey const &)
      : m_nnode(nnode), m_nface(nface), m_ncell(ncell), m_nbound(nbound)
      , m_ngstnode(0), m_ngstface(0), m_ngstcell(0)
      , m_ndcrd(std::vector<size_t>{nnode, NDIM}, 0)
      , m_fccnd(std::vector<size_t>{nface, NDIM}, 0)
      , m_fcnml(std::vector<size_t>{nface, NDIM}, 0)
      , m_fcara(std::vector<size_t>{nface}, 0)
      , m_clcnd(std::vector<size_t>{ncell, NDIM}, 0)
      , m_clvol(std::vector<size_t>{ncell}, 0)
      , m_fctpn(std::vector<size_t>{nface})
      , m_cltpn(std::vector<size_t>{ncell})
      , m_clgrp(std::vector<size_t>{ncell})
      , m_fcnds(std::vector<size_t>{nface, FCMND+1})
      , m_fccls(std::vector<size_t>{nface, FCNCL})
      , m_clnds(std::vector<size_t>{ncell, CLMND+1})
      , m_clfcs(std::vector<size_t>{ncell, CLMFC+1})
    {}
    StaticMeshBase() = delete;
    StaticMeshBase(StaticMeshBase const & ) = delete;
    StaticMeshBase(StaticMeshBase       &&) = delete;
    StaticMeshBase & operator=(StaticMeshBase const & ) = delete;
    StaticMeshBase & operator=(StaticMeshBase       &&) = delete;
    ~StaticMeshBase() = default;

private:

    static size_t calc_max_nface(SimpleArray<int_type> const & cltpn)
    {
        size_t max_nfc = 0;
        for (size_t it = 0 ; it < cltpn.nbody() ; ++it)
        {
            // FIXME: This switch is sub-optimal.
            switch (cltpn(it))
            {
            case 1:
                max_nfc += PointCellType().nface();
                break;
            case 2:
                max_nfc += LineCellType().nface();
                break;
            case 3:
                max_nfc += QuadrilateralCellType().nface();
                break;
            case 4:
                max_nfc += TriangleCellType().nface();
                break;
            case 5:
                max_nfc += HexahedronCellType().nface();
                break;
            case 6:
                max_nfc += TetrahedronCellType().nface();
                break;
            case 7:
                max_nfc += PrismCellType().nface();
                break;
            case 8:
                max_nfc += PyramidCellType().nface();
                break;
            default:
                break;
            }
        }
        return max_nfc;
    }

    void build_faces_from_cells();
    void calc_metric();

public:

    void build_interior(bool do_metric)
    {
        build_faces_from_cells();
        if (do_metric)
        {
            calc_metric();
        }
    }

    uint_type nnode() const { return m_nnode; }
    uint_type nface() const { return m_nface; }
    uint_type ncell() const { return m_ncell; }
    uint_type nbound() const { return m_nbound; }
    uint_type ngstnode() const { return m_ngstnode; }
    uint_type ngstface() const { return m_ngstface; }
    uint_type ngstcell() const { return m_ngstcell; }
    bool use_incenter() const { return m_use_incenter; }

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
}; /* end class StaticMesh2d */

class StaticMesh3d
  : public StaticMeshBase<StaticMesh3d, 3>
{
    using StaticMeshBase::StaticMeshBase;
}; /* end class StaticMesh3d */

/**
 * Extract interier faces from node list of cells.  Subroutine is designed to
 * handle all types of cells.
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::build_faces_from_cells()
{
    size_t const mface = calc_max_nface(cltpn());
    int_type computed_nface = -1;

    // create temporary tables.
    SimpleArray<int_type> tclfcs(small_vector<size_t>{ncell(), CLMFC+1}, -1);
    SimpleArray<int_type> tfctpn(small_vector<size_t>{mface}, -1);
    SimpleArray<int_type> tfcnds(small_vector<size_t>{mface, FCMND+1}, -1);
    SimpleArray<int_type> tfccls(small_vector<size_t>{mface, FCNCL}, -1);
    int_type * lclfcs = tclfcs.body();
    int_type * lfctpn = tfctpn.body();
    int_type * lfcnds = tfcnds.body();
    int_type * lfccls = tfccls.body();

    // extract face definition from the node list of cells.
    int_type * pcltpn = cltpn().body();
    int_type * pclnds = clnds().body();
    int_type * pclfcs = lclfcs;
    int_type * pfctpn = lfctpn;
    int_type * pfcnds = lfcnds;
    int_type ifc = 0;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        int_type const tpnicl = pcltpn[0];
        // parse each type of cell.
        if (tpnicl == 0 || tpnicl == 1)
        {
            // do nothing.
        }
        else if (tpnicl == 2) // line.
        {
            // extract 2 points from a line.
            pclfcs[0] = 2;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 0; // face type is point.
                pfcnds[it*(FCMND+1)] = 1;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 3) // quadrilateral.
        {
            // extract 4 lines from a quadrilateral.
            pclfcs[0] = 4;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 4) // triangle.
        {
            // extract 3 lines from a triangle.
            pclfcs[0] = 3;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 5) // hexahedron.
        {
            // extract 6 quadrilaterals from a hexahedron.
            pclfcs[0] = 6;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[6];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[5];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[8];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 6.
            pclfcs[6] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[7];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 6) // tetrahedron.
        {
            // extract 4 triangles from a tetrahedron.
            pclfcs[0] = 4;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 7) // prism.
        {
            // extract 2 triangles and 3 quadrilaterals from a prism.
            pclfcs[0] = 5;
            for (int_type it = 0 ; it < 2 ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            for (int_type it = 2 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[5];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 8) // pyramid.
        {
            // extract 4 triangles and 1 quadrilaterals from a pyramid.
            pclfcs[0] = 5;
            for (int_type it = 0 ; it < 4 ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            for (int_type it = 4 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        // advance pointers.
        pcltpn += 1;
        pclnds += CLMND+1;
        pclfcs += CLMFC+1;
    };

    // build the hash table, to know what faces connect to each node.
    /// first pass: get the maximum number of faces.
    std::vector<int_type> ndnfc(nnode());
    for (size_t ind = 0 ; ind < nnode() ; ++ind) // initialize.
    {
        ndnfc[ind] = 0;
    }
    pfcnds = lfcnds; // count.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type inf = 1 ; inf <= pfcnds[0] ; ++inf)
        {
            int_type const ind = pfcnds[inf];  // node of interest.
            ndnfc[ind] += 1;    // increment counting.
        };
        // advance pointers.
        pfcnds += FCMND+1;
    }
    int_type ndmfc = 0;  // get maximum.
    for (size_t ind = 0 ; ind < nnode() ; ++ind)
    {
        if (ndnfc[ind] > ndmfc)
        {
            ndmfc = ndnfc[ind];
        }
    }
    assert(ndmfc >= 0); // FIXME: Throw an exception.
    /// second pass: scan again to build hash table.
    SimpleArray<int_type> ndfcs(small_vector<size_t>{nnode(), static_cast<size_t>(ndmfc)+1});
    for (size_t ind = 0 ; ind < nnode() ; ++ind)
    {
        ndfcs(ind, 0) = 0;
        for (int_type it = 1 ; it <= ndmfc ; ++it)
        {
            ndfcs(ind, it) = -1;
        }
    }
    pfcnds = lfcnds; // build hash table mapping from node to face.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type inf = 1; inf <= pfcnds[0]; ++inf)
        {
            int_type const ind = pfcnds[inf];  // node of interest.
            ndfcs(ind, 0) += 1;
            ndfcs(ind, ndfcs(ind, 0)) = ifc;
        }
        // advance pointers.
        pfcnds += FCMND+1;
    }

    // scan for duplicated faces and build duplication map.
    std::vector<size_t> map(mface);
    for (size_t ifc = 0 ; ifc < mface ; ++ifc) // initialize.
    {
        map[ifc] = ifc;
    }
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        if (map[ifc] == ifc)
        {
            int_type * pifcnds = lfcnds + ifc*(FCMND+1);
            int_type const nd1 = pifcnds[1];    // take only the FIRST node of a face.
            for (int_type it = 1 ; it <= ndfcs(nd1, 0) ; ++it)
            {
                size_t const jfc = ndfcs(nd1, it);
                // test for duplication.
                if ((jfc != ifc) && (lfctpn[jfc] == lfctpn[ifc]))
                {
                    int_type * pjfcnds = lfcnds + jfc*(FCMND+1);
                    int_type cond = pjfcnds[0];
                    // scan all nodes in ifc and jfc to see if all the same.
                    for (int_type jnf = 1 ; jnf <= pjfcnds[0] ; ++jnf)
                    {
                        for (int_type inf = 1 ; inf <= pifcnds[0] ; ++inf)
                        {
                            if (pjfcnds[jnf] == pifcnds[inf])
                            {
                                cond -= 1;
                                break;
                            }
                        }
                    }
                    if (cond == 0)
                    {
                        map[jfc] = ifc;  // record duplication.
                    }
                }
            }
        }
    }

    // use the duplication map to remap nodes in faces, and build renewed map.
    std::vector<int_type> map2(mface);
    int_type * pifcnds = lfcnds;
    int_type * pjfcnds = lfcnds;
    int_type * pifctpn = lfctpn;
    int_type * pjfctpn = lfctpn;
    int_type jfc = 0;
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        if (map[ifc] == ifc)
        {
            for (int_type inf = 0 ; inf <= FCMND ; ++inf)
            {
                pjfcnds[inf] = pifcnds[inf];
            }
            pjfctpn[0] = pifctpn[0];
            map2[ifc] = jfc;
            // increment j-face.
            jfc += 1;
            pjfcnds += FCMND+1;
            pjfctpn += 1;
        }
        else
        {
            map2[ifc] = map2[map[ifc]];
        }
        // advance pointers;
        pifcnds += FCMND+1;
        pifctpn += 1;
    }
    computed_nface = jfc;    // record deduplicated number of face.

    // rebuild faces in cells and build face neighboring, according to the
    // renewed face map.
    int_type * pfccls = lfccls; // initialize.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type it = 0 ; it < FCREL ; ++it)
        {
            pfccls[it] = -1;
        }
        // advance pointers;
        pfccls += FCREL;
    }
    pclfcs = lclfcs;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        for (int_type ifl = 1 ; ifl <= pclfcs[0] ; ++ifl)
        {
            int_type const ifc = pclfcs[ifl];
            int_type const jfc = map2[ifc];
            // rebuild faces in cells.
            pclfcs[ifl] = jfc;
            // build face neighboring.
            pfccls = lfccls + jfc*FCREL;
            if (pfccls[0] == -1)
            {
                pfccls[0] = icl;
            }
            else if (pfccls[1] == -1)
            {
                pfccls[1] = icl;
            }
        }
        // advance pointers;
        pclfcs += CLMFC+1;
    };

    // recreate member tables.
    m_nface = computed_nface;
    m_fctpn.swap(SimpleArray<int_type>(small_vector<size_t>{nface()}));
    m_fcnds.swap(SimpleArray<int_type>(small_vector<size_t>{nface(), FCMND+1}));
    m_fccls.swap(SimpleArray<int_type>(small_vector<size_t>{nface(), FCNCL}));
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        for (size_t it = 0 ; it < clfcs().shape(1) ; ++it)
        {
            clfcs()(icl, it) = tclfcs(icl, it);
        }
    }
    for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
    {
        fctpn()(ifc) = tfctpn(ifc);
        for (size_t it = 0 ; it < fcnds().shape(1) ; ++it)
        {
            fcnds()(ifc, it) = tfcnds(ifc, it);
        }
        for (size_t it = 0 ; it < fccls().shape(1) ; ++it)
        {
            fccls()(ifc, it) = tfccls(ifc, it);
        }
    }
    m_fccnd.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcnml.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcara.swap(SimpleArray<real_type>(small_vector<size_t>{nface()}, 0));
}

/**
 * Calculate all metric information, including:
 *
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 *
 * And fcnds could be reordered.
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::calc_metric()
{
    // arrays.
    std::array<int_type, FCMND> ndstf; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<std::array<real_type, NDIM>, FCMND+2> cfd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<std::array<real_type, NDIM>, FCMND> radvec; // NOLINT(cppcoreguidelines-pro-type-member-init)

    // utilized arrays.
    real_type * lndcrd = ndcrd().body();
    int_type  * lfcnds = fcnds().body();
    int_type  * lfccls = fccls().body();
    real_type * lfccnd = fccnd().body();
    real_type * lfcnml = fcnml().body();
    real_type * lfcara = fcara().body();
    int_type  * lcltpn = cltpn().body();
    int_type  * lclnds = clnds().body();
    int_type  * lclfcs = clfcs().body();
    real_type * lclcnd = clcnd().body();
    real_type * lclvol = clvol().body();

    // compute face centroids.
    int_type * pfcnds = lfcnds;
    real_type * pfccnd = lfccnd;
    if (NDIM == 2)
    {
        // 2D faces must be edge.
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // point 1.
            int_type ind = pfcnds[1];
            real_type const * pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] = pndcrd[0];
            pfccnd[1] = pndcrd[1];
            // point 2.
            ind = pfcnds[2];
            pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] += pndcrd[0];
            pfccnd[1] += pndcrd[1];
            // average.
            pfccnd[0] /= 2;
            pfccnd[1] /= 2;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
        }
    }
    else if (NDIM == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            size_t const nnd = pfcnds[0];
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                int_type ind = pfcnds[inf];
                real_type const * pndcrd = lndcrd + ind*NDIM;
                cfd[inf][0]  = pndcrd[0];
                cfd[0  ][0] += pndcrd[0];
                cfd[inf][1]  = pndcrd[1];
                cfd[0  ][1] += pndcrd[1];
                cfd[inf][2]  = pndcrd[2];
                cfd[0  ][2] += pndcrd[2];
            }
            cfd[nnd+1][0] = cfd[1][0];
            cfd[nnd+1][1] = cfd[1][1];
            cfd[nnd+1][2] = cfd[1][2];
            cfd[0][0] /= nnd;
            cfd[0][1] /= nnd;
            cfd[0][2] /= nnd;
            // calculate area.
            real_type voc = 0.0;
            pfccnd[0] = pfccnd[1] = pfccnd[2] = 0.0;
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                crd[0] = (cfd[0][0] + cfd[inf][0] + cfd[inf+1][0])/3;
                crd[1] = (cfd[0][1] + cfd[inf][1] + cfd[inf+1][1])/3;
                crd[2] = (cfd[0][2] + cfd[inf][2] + cfd[inf+1][2])/3;
                real_type const du0 = cfd[inf][0] - cfd[0][0];
                real_type const du1 = cfd[inf][1] - cfd[0][1];
                real_type const du2 = cfd[inf][2] - cfd[0][2];
                real_type const dv0 = cfd[inf+1][0] - cfd[0][0];
                real_type const dv1 = cfd[inf+1][1] - cfd[0][1];
                real_type const dv2 = cfd[inf+1][2] - cfd[0][2];
                real_type const dw0 = du1*dv2 - du2*dv1;
                real_type const dw1 = du2*dv0 - du0*dv2;
                real_type const dw2 = du0*dv1 - du1*dv0;
                real_type vob = sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
                pfccnd[0] += crd[0] * vob;
                pfccnd[1] += crd[1] * vob;
                pfccnd[2] += crd[2] * vob;
                voc += vob;
            }
            pfccnd[0] /= voc;
            pfccnd[1] /= voc;
            pfccnd[2] /= voc;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
        }
    }

    // compute face normal vector and area.
    pfcnds = lfcnds;
    pfccnd = lfccnd;
    if (NDIM == 2)
    {
        real_type * pfcnml = lfcnml;
        real_type * pfcara = lfcara;
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // 2D faces are always lines.
            real_type const * pndcrd = lndcrd + pfcnds[1]*NDIM;
            real_type const * p2ndcrd = lndcrd + pfcnds[2]*NDIM;
            // face normal.
            pfcnml[0] = p2ndcrd[1] - pndcrd[1];
            pfcnml[1] = -(p2ndcrd[0] - pndcrd[0]);
            // face ara.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]);
            // normalize face normal.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            // advance pointers.
            pfcnds += FCMND+1;
            pfcnml += NDIM;
            pfcara += 1;
        }
    }
    else if (NDIM == 3)
    {
        real_type * pfcnml = lfcnml;
        real_type * pfcara = lfcara;
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // compute radial vector.
            size_t const nnd = pfcnds[0];
            for (size_t inf = 0 ; inf < nnd ; ++inf)
            {
                int_type ind = pfcnds[inf+1];
                real_type const * pndcrd = lndcrd + ind*NDIM;
                radvec[inf][0] = pndcrd[0] - pfccnd[0];
                radvec[inf][1] = pndcrd[1] - pfccnd[1];
                radvec[inf][2] = pndcrd[2] - pfccnd[2];
            }
            // compute cross product.
            pfcnml[0] = radvec[nnd-1][1]*radvec[0][2]
                      - radvec[nnd-1][2]*radvec[0][1];
            pfcnml[1] = radvec[nnd-1][2]*radvec[0][0]
                      - radvec[nnd-1][0]*radvec[0][2];
            pfcnml[2] = radvec[nnd-1][0]*radvec[0][1]
                      - radvec[nnd-1][1]*radvec[0][0];
            for (size_t ind = 1 ; ind < nnd ; ++ind)
            {
                pfcnml[0] += radvec[ind-1][1]*radvec[ind][2]
                           - radvec[ind-1][2]*radvec[ind][1];
                pfcnml[1] += radvec[ind-1][2]*radvec[ind][0]
                           - radvec[ind-1][0]*radvec[ind][2];
                pfcnml[2] += radvec[ind-1][0]*radvec[ind][1]
                           - radvec[ind-1][1]*radvec[ind][0];
            }
            // compute face area.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]
                           + pfcnml[2]*pfcnml[2]);
            // normalize normal vector.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            pfcnml[2] /= pfcara[0];
            // get real face area.
            pfcara[0] /= 2.0;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
            pfcnml += NDIM;
            pfcara += 1;
        }
    }

    // compute cell centroids.
    int_type * pclnds = lclnds;
    int_type * pclfcs = lclfcs;
    if (NDIM == 2)
    {
        real_type * pclcnd = lclcnd;
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (lcltpn[icl] == 3))
            {
                real_type const * pndcrd = lndcrd + pclnds[1]*NDIM;
                real_type vob = lfcara[pclfcs[2]];
                real_type voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            }
            else
            {
                // averaged point.
                crd[0] = crd[1] = 0.0;
                size_t const nnd = pclnds[0];
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = pclnds[inc];
                    real_type const * pndcrd = lndcrd + ind*NDIM;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                pclcnd[0] = pclcnd[1] = 0.0;
                size_t const nfc = pclfcs[0];
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type ifc = pclfcs[ifl];
                    pfccnd = lfccnd + ifc*NDIM;
                    real_type const * pfcnml = lfcnml + ifc*NDIM;
                    real_type const * pfcara = lfcara + ifc;
                    real_type const du0 = crd[0] - pfccnd[0];
                    real_type const du1 = crd[1] - pfccnd[1];
                    real_type vob = fabs(du0*pfcnml[0] + du1*pfcnml[1]) * pfcara[0];
                    voc += vob;
                    real_type const dv0 = pfccnd[0] + du0/3;
                    real_type const dv1 = pfccnd[1] + du1/3;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                }
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            }
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += NDIM;
        }
    }
    else if (NDIM == 3)
    {
        real_type * pclcnd = lclcnd;
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (lcltpn[icl] == 5))
            {
                real_type const * pndcrd = lndcrd + pclnds[1]*NDIM;
                real_type vob = lfcara[pclfcs[4]];
                real_type voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pclcnd[2] = vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[2]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[4]*NDIM;
                vob = lfcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            }
            else
            {
                // averaged point.
                crd[0] = crd[1] = crd[2] = 0.0;
                size_t const nnd = pclnds[0];
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = pclnds[inc];
                    real_type const * pndcrd = lndcrd + ind*NDIM;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                    crd[2] += pndcrd[2];
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                crd[2] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                pclcnd[0] = pclcnd[1] = pclcnd[2] = 0.0;
                size_t const nfc = pclfcs[0];
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type ifc = pclfcs[ifl];
                    pfccnd = lfccnd + ifc*NDIM;
                    real_type const * pfcnml = lfcnml + ifc*NDIM;
                    real_type const * pfcara = lfcara + ifc;
                    real_type const du0 = crd[0] - pfccnd[0];
                    real_type const du1 = crd[1] - pfccnd[1];
                    real_type const du2 = crd[2] - pfccnd[2];
                    real_type vob = fabs(du0*pfcnml[0] + du1*pfcnml[1] + du2*pfcnml[2]) * pfcara[0];
                    voc += vob;
                    real_type const dv0 = pfccnd[0] + du0/4;
                    real_type const dv1 = pfccnd[1] + du1/4;
                    real_type const dv2 = pfccnd[2] + du2/4;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                    pclcnd[2] += dv2 * vob;
                }
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            }
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += NDIM;
        }
    }

    // compute volume for each cell.
    pclfcs = lclfcs;
    real_type const * pclcnd = lclcnd;
    real_type * pclvol = lclvol;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        pclvol[0] = 0.0;
        size_t const nfc = pclfcs[0];
        for (size_t it = 1 ; it <= nfc ; ++it)
        {
            int_type ifc = pclfcs[it];
            int_type * pfccls = lfccls + ifc*FCREL;
            pfcnds = lfcnds + ifc*(FCMND+1);
            pfccnd = lfccnd + ifc*NDIM;
            real_type * pfcnml = lfcnml + ifc*NDIM;
            real_type const * pfcara = lfcara + ifc;
            // calculate volume associated with each face.
            real_type vol = 0.0;
            for (size_t idm = 0 ; idm < NDIM ; ++idm)
            {
                vol += (pfccnd[idm] - pclcnd[idm]) * pfcnml[idm];
            }
            vol *= pfcara[0];
            // check if need to reorder node definition and connecting cell
            // list for the face.
            size_t const this_fcl = pfccls[0];
            if (vol < 0.0)
            {
                if (this_fcl == icl)
                {
                    size_t const nnd = pfcnds[0];
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        ndstf[jt] = pfcnds[nnd-jt];
                    }
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        pfcnds[jt+1] = ndstf[jt];
                    }
                    for (size_t idm = 0 ; idm < NDIM ; ++idm)
                    {
                        pfcnml[idm] = -pfcnml[idm];
                    }
                }
                vol = -vol;
            }
            else
            {
                if (this_fcl != icl)
                {
                    size_t const nnd = pfcnds[0];
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        ndstf[jt] = pfcnds[nnd-jt];
                    }
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        pfcnds[jt+1] = ndstf[jt];
                    }
                    for (size_t idm = 0 ; idm < NDIM ; ++idm)
                    {
                        pfcnml[idm] = -pfcnml[idm];
                    }
                }
            }
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        }
        // calculate the real volume.
        pclvol[0] /= NDIM;
        // advance pointers.
        pclfcs += CLMFC+1;
        pclcnd += NDIM;
        pclvol += 1;
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
