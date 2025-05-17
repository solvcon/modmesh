#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <functional>

#include <modmesh/base.hpp>
#include <modmesh/mesh/mesh.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/inout/inout_util.hpp>

namespace modmesh
{
namespace inout
{
struct GmshElementDef
{
    static GmshElementDef by_id(uint16_t id);

    GmshElementDef(uint8_t ndim, uint16_t nnds, uint8_t mmtpn, small_vector<uint8_t> const & mmcl)
        : m_ndim(ndim)
        , m_nnds(nnds)
        , m_mmtpn(mmtpn)
        , m_mmcl(mmcl)
    {
    }

    ~GmshElementDef() = default;

    GmshElementDef() = default;
    GmshElementDef(const GmshElementDef & other) = delete;
    GmshElementDef(GmshElementDef && other) = delete;
    GmshElementDef & operator=(const GmshElementDef & other) = delete;
    GmshElementDef & operator=(GmshElementDef && other) = delete;

    uint8_t ndim() const { return m_ndim; }
    uint16_t nnds() const { return m_nnds; }
    uint8_t mmtpn() const { return m_mmtpn; }
    small_vector<uint8_t> mmcl() const { return m_mmcl; }

private:
    uint8_t m_ndim = 0; /* Number of dimension */
    uint16_t m_nnds = 0; /* Number of nodes     */
    uint8_t m_mmtpn = 0; /* modmesh cell type   */
    small_vector<uint8_t> m_mmcl; /* modmesh cell order  */
}; /* end struct GmshElementDef */

class Gmsh
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Gmsh(const std::string & data);

    ~Gmsh() = default;

    Gmsh() = delete;
    Gmsh(Gmsh const & other) = delete;
    Gmsh(Gmsh && other) = delete;
    Gmsh & operator=(Gmsh const & other) = delete;
    Gmsh & operator=(Gmsh && other) = delete;

    std::shared_ptr<StaticMesh> to_block(void);

private:
    enum class FormatState
    {
        BEGIN,
        META_END,
        PYHSICAL_NAME_END,
        NODE_END,
        ELEMENT_END
    };

    bool is_valid_transition(const std::string s);
    void load_meta(void);
    void load_physical(void);
    void load_nodes(void);
    void load_elements(void);
    void build_interior(const std::shared_ptr<StaticMesh> & blk);

    std::stringstream stream;
    FormatState last_fmt_state = FormatState::BEGIN;

    real_type msh_ver = 0.0;

    uint_type msh_file_type = 0;
    uint_type msh_data_size = 0;
    uint_type nnodes = 0;

    SimpleArray<int_type> m_cltpn;

    SimpleArray<real_type> m_nds;

    SimpleArray<uint_type> m_elgrp;
    SimpleArray<uint_type> m_elgeo;
    SimpleArray<uint_type> m_eldim;
    SimpleArray<uint_type> m_usnds;
    SimpleArray<uint_type> m_ndmap;

    std::unordered_map<uint_type, small_vector<uint_type>> m_elems;
}; /* end class Gmsh */

inline GmshElementDef GmshElementDef::by_id(uint16_t id)
{
#define VEC(...) __VA_ARGS__
#define MM_DECL_SWITCH_ELEMENT_TYPE(ID, NDIM, NNDS, MMTPN, MMCL) \
    case ID: return GmshElementDef(NDIM, NNDS, MMTPN, small_vector<uint8_t>{MMCL}); break;
    switch (id)
    {
        // clang-format off
        // NOLINTBEGIN(bugprone-branch-clone,-warnings-as-error)
        //                          id, dim, nnodes, cell type, cell order
        MM_DECL_SWITCH_ELEMENT_TYPE( 1,   1,      2,         2, VEC(0, 1))                       // 2-node line
        MM_DECL_SWITCH_ELEMENT_TYPE( 2,   2,      3,         4, VEC(0, 1, 2))                    // 3-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE( 3,   2,      4,         3, VEC(0, 1, 2, 3))                 // 4-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE( 4,   3,      4,         6, VEC(0, 1, 2, 3))                 // 4-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE( 5,   3,      8,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 8-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE( 6,   3,      6,         7, VEC(0, 2, 1, 3, 5, 4))           // 6-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE( 7,   3,      5,         8, VEC(0, 1, 2, 3, 4))              // 5-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE( 8,   1,      3,         2, VEC(0, 1))                       // 3-node line
        MM_DECL_SWITCH_ELEMENT_TYPE( 9,   2,      6,         4, VEC(0, 1, 2))                    // 6-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(10,   2,      9,         3, VEC(0, 1, 2, 3))                 // 9-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE(11,   3,     10,         6, VEC(0, 1, 2, 3))                 // 10-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(12,   3,     27,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 27-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(13,   3,     18,         7, VEC(0, 2, 1, 3, 5, 4))           // 18-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE(14,   3,     14,         8, VEC(0, 1, 2, 3, 4))              // 14-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE(15,   0,      1,         1, VEC(0))                          // 1-node point
        MM_DECL_SWITCH_ELEMENT_TYPE(16,   2,      8,         3, VEC(0, 1, 2, 3))                 // 8-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE(17,   3,     20,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 20-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(18,   3,     15,         7, VEC(0, 2, 1, 3, 5, 4))           // 15-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE(19,   3,     13,         8, VEC(0, 1, 2, 3, 4))              // 13-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE(20,   2,      9,         4, VEC(0, 1, 2))                    // 9-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(21,   2,     10,         4, VEC(0, 1, 2))                    // 10-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(22,   2,     12,         4, VEC(0, 1, 2))                    // 12-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(23,   2,     15,         4, VEC(0, 1, 2))                    // 15-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(24,   2,     15,         4, VEC(0, 1, 2))                    // 15-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(25,   2,     21,         4, VEC(0, 1, 2))                    // 21-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(26,   1,      4,         2, VEC(0, 1))                       // 4-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(27,   1,      5,         2, VEC(0, 1))                       // 5-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(28,   1,      6,         2, VEC(0, 1))                       // 6-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(29,   3,     20,         6, VEC(0, 1, 2, 3))                 // 20-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(30,   3,     35,         6, VEC(0, 1, 2 ,3))                 // 35-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(31,   3,     56,         6, VEC(0, 1, 2, 3))                 // 56-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(92,   3,     64,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 64-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(93,   3,    125,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 125-node hexahedron
        default: return GmshElementDef{}; break;
        // NOLINTEND(bugprone-branch-clone,-warnings-as-error)
        // clang-format on
    }
#undef MM_DECL_SWITCH_ELEMENT_TYPE
#undef VEC
}
} /* end namespace inout */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
