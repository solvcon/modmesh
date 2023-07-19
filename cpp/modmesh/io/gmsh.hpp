#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <functional>

#include <modmesh/base.hpp>
#include <modmesh/mesh/mesh.hpp>
#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{
namespace IO
{
namespace Gmsh
{
struct ElementDef
{
    static ElementDef by_id(uint16_t id);

    ElementDef(uint8_t ndim, uint16_t nnds, uint8_t mmtpn, small_vector<uint8_t> const & mmcl)
        : m_ndim(ndim)
        , m_nnds(nnds)
        , m_mmtpn(mmtpn)
        , m_mmcl(mmcl)
    {
    }

    ~ElementDef() = default;

    ElementDef() = default;
    ElementDef(const ElementDef & other) = delete;
    ElementDef(ElementDef && other) = delete;
    ElementDef & operator=(const ElementDef & other) = delete;
    ElementDef & operator=(ElementDef && other) = delete;

    uint8_t ndim() const { return m_ndim; }
    uint16_t nnds() const { return m_nnds; }
    uint8_t mmtpn() const { return m_mmtpn; }
    small_vector<uint8_t> mmcl() const { return m_mmcl; }

private:
    uint8_t m_ndim = 0; /* Number of dimension */
    uint16_t m_nnds = 0; /* Number of nodes     */
    uint8_t m_mmtpn = 0; /* modmesh cell type   */
    small_vector<uint8_t> m_mmcl; /* modmesh cell order  */
}; /* end struct ElementDef */

inline ElementDef ElementDef::by_id(uint16_t id)
{
#define VEC(...) __VA_ARGS__
#define MM_DECL_SWITCH_ELEMENT_TYPE(ID, NDIM, NNDS, MMTPN, MMCL) \
    case ID: return ElementDef(NDIM, NNDS, MMTPN, small_vector<uint8_t>{MMCL}); break;
    switch (id)
    {
        // clang-format off
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
        default: return ElementDef{}; break;
        // clang-format on
    }
#undef MM_DECL_SWITCH_ELEMENT_TYPE
#undef VEC
}

class Gmsh
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Gmsh(const std::string & file_path)
    {
        bool meta_enter = false, node_enter = false, element_enter = false;
        // clang-format off
        std::unordered_map<std::string, std::function<void()>> keyword_handler = {
            {"$MeshFormat", [this, &meta_enter]() { _load_meta(); meta_enter = true; }},
            {"$Nodes", [this, &node_enter]() { _load_nodes(); node_enter = true; }},
            {"$Elements", [this, &element_enter]() { _load_elements(); element_enter = true; }},
            {"$PhysicalNames", [this]() { _load_physical(); }}
        };

        stream.open(file_path, std::ios::in);
        if (!stream.good())
        {
            throw std::invalid_argument(Formatter() << file_path << " path invalid.");
        }

        std::string line;
        while (std::getline(stream, line))
        {
            // Using a finite state machine to check the input msh file format is valid or not.
            // $ is a keyword to trigger state transition.
            if (line.find('$') != std::string::npos)
            {
                auto it = keyword_handler.find(line);
                if (it != keyword_handler.end())
                {
                    if (_isValidTransition(it->first))
                    {
                        it->second();
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
        }

        // MeshFormat, Nodes and Elements section is mandatory in gmsh msh file,
        // therefore need to check these sections are exist or not otherwise
        // modmesh may crash during mesh processing.
        if (!(meta_enter && node_enter && element_enter))
        {
            throw std::invalid_argument("Incorrect msh file format.");
        }
    }

    ~Gmsh()
    {
        stream.close();
    }

    Gmsh() = delete;
    Gmsh(Gmsh const & other) = delete;
    Gmsh(Gmsh && other) = delete;
    Gmsh & operator=(Gmsh const & other) = delete;
    Gmsh & operator=(Gmsh && other) = delete;

    std::shared_ptr<StaticMesh> toblock(void);

private:
    enum class FormatState
    {
        BEGIN,
        META_END,
        PYHSICAL_NAME_END,
        NODE_END,
        ELEMENT_END
    };

    // Check the finite state machine state transition is valid or not to check msh file format is correct
    bool _isValidTransition(const std::string s)
    {
        if (last_fmt_state == FormatState::BEGIN)
        {
            return s == "$MeshFormat";
        }
        else if (last_fmt_state == FormatState::META_END || last_fmt_state == FormatState::PYHSICAL_NAME_END)
        {
            return s == "$PhysicalNames" || s == "$Nodes";
        }
        else if (last_fmt_state == FormatState::NODE_END)
        {
            return s == "$Elements";
        }

        return false;
    }

    void _load_meta(void)
    {
        std::string line;

        while (std::getline(stream, line) && line.find('$') == std::string::npos)
        {
            auto tokens = _tokenize(line, ' ');
            msh_ver = std::stod(tokens[0]);

            // The parse only support ver 2.2 msh file.
            if (msh_ver != 2.2)
            {
                throw std::invalid_argument(Formatter() << "modmesh does not support msh file ver " << msh_ver << ".");
            }

            msh_file_type = std::stoi(tokens[1]);
            msh_data_size = std::stoi(tokens[2]);
        }

        if (!line.compare("$EndMeshFormat"))
        {
            last_fmt_state = FormatState::META_END;
        }
    }

    // TODO: PhysicalNames section parsing logic not complete yet, but without PhysicalNames section
    //       modmesh mesh viewer still working, therefore we can finish this later.
    void _load_physical(void)
    {
        std::string line;

        while (std::getline(stream, line) && line.find('$') == std::string::npos) {}

        if (!line.compare("$EndPhysicalNames"))
        {
            last_fmt_state = FormatState::PYHSICAL_NAME_END;
        }
    }

    void _load_nodes(void)
    {
        std::string line;
        std::getline(stream, line);
        auto nnode = std::stoul(line);

        m_nds.remake(small_vector<size_t>{nnode, 3}, 0);

        while (std::getline(stream, line) && line.find('$') == std::string::npos)
        {
            auto tokens = _tokenize(line, ' ');
            // gmsh node index is 1-based index
            m_nds(std::stoul(tokens[0]) - 1, 0) = std::stod(tokens[1]);
            m_nds(std::stoul(tokens[0]) - 1, 1) = std::stod(tokens[2]);
            m_nds(std::stoul(tokens[0]) - 1, 2) = std::stod(tokens[3]);
        }

        if (!line.compare("$EndNodes"))
        {
            last_fmt_state = FormatState::NODE_END;
        }
    }

    void _load_elements(void)
    {
        std::string line;
        std::getline(stream, line);
        auto nelement = std::stoul(line);
        std::vector<uint_type> usnds;

        m_cltpn.remake(small_vector<size_t>{nelement}, 0);
        m_elgrp.remake(small_vector<size_t>{nelement}, 0);
        m_elgeo.remake(small_vector<size_t>{nelement}, 0);
        m_eldim.remake(small_vector<size_t>{nelement}, 0);

        uint_type idx = 0;

        while (std::getline(stream, line) && line.find('$') == std::string::npos && idx < nelement)
        {
            auto tokens = _tokenize(line, ' ');

            // parse element type
            auto tpn = std::stoul(tokens[1]);
            auto eldef = ElementDef::by_id(tpn);
            // parse element tag
            std::vector<uint_type> tag;
            for (size_t i = 0; i < std::stoul(tokens[2]); ++i)
            {
                tag.push_back(std::stoul(tokens[3 + i]));
            }

            // parse node number list
            std::vector<uint_type> nds;
            for (size_t i = 3 + std::stoul(tokens[2]); i < tokens.size(); ++i)
            {
                nds.push_back(std::stoul(tokens[i]));
            }

            m_cltpn[idx] = eldef.mmtpn();
            m_elgrp[idx] = tag[0];
            m_elgeo[idx] = tag[1];
            m_eldim[idx] = eldef.ndim();

            small_vector<uint_type> nds_temp(nds.size() + 1, 0);
            auto mmcl = eldef.mmcl();

            for (size_t i = 0; i < mmcl.size(); ++i)
            {
                nds_temp[mmcl[i] + 1] = nds[i] - 1;
            }
            usnds.insert(usnds.end(), nds_temp.begin() + 1, nds_temp.end());
            nds_temp[0] = mmcl.size();
            m_elems.insert(std::pair{idx, nds_temp});
            idx++;
        }

        if (!line.compare("$EndElements"))
        {
            last_fmt_state = FormatState::ELEMENT_END;
        }

        // sorting used node and remove duplicate node id
        std::sort(usnds.begin(), usnds.end());
        auto remove = std::unique(usnds.begin(), usnds.end());
        usnds.erase(remove, usnds.end());

        // put used node id to m_ndmap
        m_ndmap.remake(small_vector<size_t>{usnds.size()}, -1);
        for (size_t i = 0; i < usnds.size(); ++i)
        {
            m_ndmap(usnds[i]) = i;
        }
    }

    std::vector<std::string> _tokenize(const std::string & str, const char delim)
    {
        std::vector<std::string> output;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, delim))
        {
            output.push_back(token);
        }
        return output;
    }

    void _build_interior(std::shared_ptr<StaticMesh> blk)
    {
        blk->cltpn().swap(m_cltpn);
        blk->ndcrd().swap(m_nds);
        for (size_t i = 0; i < m_elems.size(); ++i)
        {
            blk->clnds()(i, 0) = m_elems[i][0];
            for (size_t j = 1; j <= m_elems[i][0]; ++j)
            {

                blk->clnds()(i, j) = m_elems[i][j];
            }
        }
        blk->build_interior(true);
        blk->build_boundary();
        blk->build_ghost();
    }

    std::ifstream stream;
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

inline std::shared_ptr<StaticMesh> Gmsh::toblock(void)
{
    std::shared_ptr<StaticMesh> block = StaticMesh::construct(m_eldim.max(), m_nds.shape(0), 0, m_elems.size());
    _build_interior(block);
    return block;
}
} /* end namespace Gmsh */
} /* end namespace IO */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
