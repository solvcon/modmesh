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
#include <modmesh/buffer/buffer.hpp>

#include <cmath>
#include <vector>
#include <numeric>

namespace modmesh
{

/**
 * Cell type for unstructured mesh.
 */
struct CellType : NumberBase<int32_t, double>
{

    /* symbols for type id codes */
    static constexpr const uint8_t NONCELLTYPE = 0; /* not a cell type */
    static constexpr const uint8_t POINT = 1;
    static constexpr const uint8_t LINE = 2;
    static constexpr const uint8_t QUADRILATERAL = 3;
    static constexpr const uint8_t TRIANGLE = 4;
    static constexpr const uint8_t HEXAHEDRON = 5;
    static constexpr const uint8_t TETRAHEDRON = 6;
    static constexpr const uint8_t PRISM = 7;
    static constexpr const uint8_t PYRAMID = 8;
    /* number of all types; the same as the last type id code */
    static constexpr const uint8_t NTYPE = 8;

    //< Maximum number of nodes in a face.
    static constexpr const uint8_t FCNND_MAX = 4;
    //< Maximum number of nodes in a cell.
    static constexpr const uint8_t CLNND_MAX = 8;
    //< Maximum number of faces in a cell.
    static constexpr const uint8_t CLNFC_MAX = 6;

    static CellType by_id(uint8_t id);

    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */
    CellType(uint8_t id_in, uint8_t ndim_in, uint8_t nnode_in, uint8_t nedge_in, uint8_t nsurface_in)
        : m_id(id_in)
        , m_ndim(ndim_in)
        , m_nnode(nnode_in)
        , m_nedge(nedge_in)
        , m_nsurface(nsurface_in)
    {
    }

    CellType()
        : CellType(NONCELLTYPE, 0, 0, 0, 0)
    {
    }

    uint8_t id() const { return m_id; }
    uint8_t ndim() const { return m_ndim; }
    uint8_t nnode() const { return m_nnode; }
    uint8_t nedge() const { return m_nedge; }
    uint8_t nsurface() const { return m_nsurface; }

    uint8_t nface() const { return 2 == m_ndim ? nedge() : nsurface(); }

    const char * name() const
    {
        switch (id())
        {
        case POINT /* 1 */: return "point"; break;
        case LINE /* 2 */: return "line"; break;
        case QUADRILATERAL /* 3 */: return "quadrilateral"; break;
        case TRIANGLE /* 4 */: return "triangle"; break;
        case HEXAHEDRON /* 5 */: return "hexahedron"; break;
        case TETRAHEDRON /* 6 */: return "tetrahedron"; break;
        case PRISM /* 7 */: return "prism"; break;
        case PYRAMID /* 8 */: return "pyramid"; break;
        case NONCELLTYPE /* 0 */:
        default /* other */: return "noncelltype"; break;
        }
    }

private:

    uint8_t m_id : 6;
    uint8_t m_ndim : 2;
    uint8_t m_nnode = 0;
    uint8_t m_nedge = 0;
    uint8_t m_nsurface = 0;

}; /* end struct CellType */

static_assert(sizeof(CellType) == 4);

inline CellType CellType::by_id(uint8_t id)
{

#define MM_DECL_SWITCH_CELL_TYPE(TYPE, NDIM, NNODE, NEDGE, NSURFACE) \
    case TYPE: return CellType(TYPE, NDIM, NNODE, NEDGE, NSURFACE); break;

    switch (id)
    {
        // clang-format off
        //                        id, ndim, nnode, nedge, nsurface
        MM_DECL_SWITCH_CELL_TYPE(  0,    0,     0,     0,        0 ) // non-type
        MM_DECL_SWITCH_CELL_TYPE(  1,    0,     1,     0,        0 ) // point/node/vertex
        MM_DECL_SWITCH_CELL_TYPE(  2,    1,     2,     0,        0 ) // line/edge
        MM_DECL_SWITCH_CELL_TYPE(  3,    2,     4,     4,        0 ) // quadrilateral
        MM_DECL_SWITCH_CELL_TYPE(  4,    2,     3,     3,        0 ) // triangle
        MM_DECL_SWITCH_CELL_TYPE(  5,    3,     8,    12,        6 ) // hexahedron/brick
        MM_DECL_SWITCH_CELL_TYPE(  6,    3,     4,     6,        4 ) // tetrahedron
        MM_DECL_SWITCH_CELL_TYPE(  7,    3,     6,     9,        5 ) // prism
        MM_DECL_SWITCH_CELL_TYPE(  8,    3,     5,     8,        5 ) // pyramid
        default: return CellType{}; break;
        // clang-format on
    }

#undef MM_DECL_SWITCH_CELL_TYPE
}

struct StaticMeshConstant
{

    static constexpr const uint8_t FCMND = CellType::FCNND_MAX;
    static constexpr const uint8_t CLMND = CellType::CLNND_MAX;
    static constexpr const uint8_t CLMFC = CellType::CLNFC_MAX;
    static constexpr const uint8_t FCREL = 4;
    static constexpr const uint8_t BFREL = 3;

}; /* end struct StaticMeshConstant */

// TODO: StaticMeshBC may use polymorphism.
class StaticMeshBC
    : public NumberBase<int32_t, double>
    , public StaticMeshConstant
{

public:

    using number_base = NumberBase<int32_t, double>;

    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

private:

    /**
     * First column is the face index in block.  The second column is the face
     * index in bndfcs.  The third column is the face index of the related
     * block (if exists).
     */
    SimpleArray<int_type> m_facn = SimpleArray<int_type>(std::vector<size_t>{0});

public:

    static const std::string & NONAME()
    {
        static const std::string str("<NONAME>");
        return str;
    }

    StaticMeshBC() = default;

    explicit StaticMeshBC(size_t nbound)
        : m_facn(SimpleArray<int_type>(std::vector<size_t>{nbound, BFREL}))
    {
    }

    StaticMeshBC(StaticMeshBC const & other)
    {
        if (this != &other)
        {
            m_facn = other.m_facn;
        }
    }

    StaticMeshBC(StaticMeshBC && other)
    {
        if (this != &other)
        {
            m_facn = std::move(other.m_facn);
        }
    }

    StaticMeshBC & operator=(StaticMeshBC const & other)
    {
        if (this != &other)
        {
            m_facn = other.m_facn;
        }
        return *this;
    }

    StaticMeshBC & operator=(StaticMeshBC && other)
    {
        if (this != &other)
        {
            m_facn = std::move(other.m_facn);
        }
        return *this;
    }

    ~StaticMeshBC() = default;

    size_t nbound() const { return m_facn.nbody(); }

    SimpleArray<int_type> const & facn() const { return m_facn; }
    SimpleArray<int_type> & facn() { return m_facn; }

}; /* end class StaticMeshBC */

class StaticMesh
    : public NumberBase<int32_t, double>
    , public StaticMeshConstant
    , public std::enable_shared_from_this<StaticMesh>
{

private:

    class ctor_passkey
    {
    };

public:

    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

    template <typename... Args>
    static std::shared_ptr<StaticMesh> construct(Args &&... args)
    {
        return std::make_shared<StaticMesh>(std::forward<Args>(args)..., ctor_passkey());
    }

    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */
    StaticMesh(uint8_t ndim, uint_type nnode, uint_type nface, uint_type ncell, ctor_passkey const &)
        : m_ndim(ndim)
        , m_nnode(nnode)
        , m_nface(nface)
        , m_ncell(ncell)
        , m_ndcrd(std::vector<size_t>{nnode, m_ndim}, 0)
        , m_fccnd(std::vector<size_t>{nface, m_ndim}, 0)
        , m_fcnml(std::vector<size_t>{nface, m_ndim}, 0)
        , m_fcara(std::vector<size_t>{nface}, 0)
        , m_clcnd(std::vector<size_t>{ncell, m_ndim}, 0)
        , m_clvol(std::vector<size_t>{ncell}, 0)
        , m_fctpn(std::vector<size_t>{nface})
        , m_cltpn(std::vector<size_t>{ncell})
        , m_clgrp(std::vector<size_t>{ncell})
        , m_fcnds(std::vector<size_t>{nface, FCMND + 1})
        , m_fccls(std::vector<size_t>{nface, FCREL})
        , m_clnds(std::vector<size_t>{ncell, CLMND + 1})
        , m_clfcs(std::vector<size_t>{ncell, CLMFC + 1})
        , m_ednds(std::vector<size_t>{0, 2})
        , m_bndfcs(std::vector<size_t>{0, StaticMeshBC::BFREL})
    {
    }
    StaticMesh() = delete;
    StaticMesh(StaticMesh const &) = delete;
    StaticMesh(StaticMesh &&) = delete;
    StaticMesh & operator=(StaticMesh const &) = delete;
    StaticMesh & operator=(StaticMesh &&) = delete;
    ~StaticMesh() = default;

public:

    uint8_t ndim() const { return m_ndim; }
    uint_type nnode() const { return m_nnode; }
    uint_type nface() const { return m_nface; }
    uint_type ncell() const { return m_ncell; }
    uint_type nbound() const { return m_nbound; }
    uint_type ngstnode() const { return m_ngstnode; }
    uint_type ngstface() const { return m_ngstface; }
    uint_type ngstcell() const { return m_ngstcell; }
    bool use_incenter() const { return m_use_incenter; }

    uint_type nedge() const { return static_cast<uint_type>(m_ednds.shape(0)); }
    size_t nbcs() const { return m_bcs.size(); }

    /**
     * Get the "self" cell number of the input face by index.  A shorthand of
     * fccls()[ifc][0] .
     *
     * @param[in] ifc index of the face of interest.
     * @return        index of the cell.
     */
    int_type fcicl(int_type ifc) const { return m_fccls(ifc, 0); }

    /**
     * Get the "related" cell number of the input face by index.  A shorthand
     * of fccls()[ifc][1] .
     *
     * @param[in] ifc index of the face of interest.
     * @return        index of the cell.
     */
    int_type fcjcl(int_type ifc) const { return m_fccls(ifc, 1); }

    // Helpers for interior data.
public:

    void build_interior(bool do_metric, bool do_edge = true)
    {
        build_faces_from_cells();
        if (do_metric)
        {
            calc_metric();
        }
        if (do_edge)
        {
            build_edge();
        }
    }

    void build_edge();

private:

    void build_faces_from_cells();
    void calc_metric();

    // Helpers for boundary data (as well as ghost).
public:

    void build_boundary();
    void build_ghost();

private:

    std::tuple<size_t, size_t, size_t> count_ghost() const;
    void fill_ghost();

    // Shape data.
private:

    uint8_t m_ndim = 0;
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
#define MM_DECL_StaticMesh_ARRAY(TYPE, NAME)                            \
public:                                                                 \
    SimpleArray<TYPE> const & NAME() const { return m_##NAME; }         \
    SimpleArray<TYPE> & NAME() { return m_##NAME; }                     \
    template <typename... Args>                                         \
    TYPE const & NAME(Args... args) const { return m_##NAME(args...); } \
    template <typename... Args>                                         \
    TYPE & NAME(Args... args) { return m_##NAME(args...); }             \
                                                                        \
private:                                                                \
    SimpleArray<TYPE> m_##NAME

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
    MM_DECL_StaticMesh_ARRAY(int_type, ednds);
    // boundary information.
    MM_DECL_StaticMesh_ARRAY(int_type, bndfcs);
    std::vector<StaticMeshBC> m_bcs;

#undef MM_DECL_StaticMesh_ARRAY

}; /* end class StaticMesh */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
