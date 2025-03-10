#include <modmesh/inout/gmsh.hpp>
#ifdef _WIN32
#include <algorithm>
#endif // _WIN32
namespace modmesh
{
namespace inout
{
Gmsh::Gmsh(const std::string & data)
{
    bool meta_enter = false;
    bool node_enter = false;
    bool element_enter = false;
    // clang-format off
    std::unordered_map<std::string, std::function<void()>> keyword_handler = {
        {"$MeshFormat", [this, &meta_enter]() { load_meta(); meta_enter = true; }},
        {"$Nodes", [this, &node_enter]() { load_nodes(); node_enter = true; }},
        {"$Elements", [this, &element_enter]() { load_elements(); element_enter = true; }},
        {"$PhysicalNames", [this]() { load_physical(); }}};
    // clang-format on

    // String stream on windows need to remove \r for keyword comparison.
    // DOS file newline character is CRLF, std::getline default using LF as delimeter
    // therefore string seperated by std::getline will contain \r, it will cause
    // keyword compare failed.
    // TODO: Keyword comparison can use regular expression.
#ifdef _WIN32
    std::string data_copy = data;
    data_copy.erase(std::remove(data_copy.begin(), data_copy.end(), '\r'), data_copy.end());
    stream << data_copy;
#else // _WIN32
    stream << data;
#endif // _WIN32

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
                if (is_valid_transition(it->first))
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

std::shared_ptr<StaticMesh> Gmsh::to_block()
{
    std::shared_ptr<StaticMesh> block = StaticMesh::construct(
        m_eldim.max(),
        static_cast<StaticMesh::uint_type>(m_nds.shape(0)),
        0,
        static_cast<StaticMesh::uint_type>(m_elems.size()));
    build_interior(block);
    return block;
}

void Gmsh::build_interior(const std::shared_ptr<StaticMesh> & blk)
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

// Check the finite state machine state transition is valid or not to check msh file format is correct
bool Gmsh::is_valid_transition(const std::string s)
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

void Gmsh::load_meta(void)
{
    std::string line;

    while (std::getline(stream, line) && line.find('$') == std::string::npos)
    {
        auto tokens = tokenize(line, ' ');
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
void Gmsh::load_physical(void)
{
    std::string line;

    while (std::getline(stream, line) && line.find('$') == std::string::npos) {}

    if (!line.compare("$EndPhysicalNames"))
    {
        last_fmt_state = FormatState::PYHSICAL_NAME_END;
    }
}

void Gmsh::load_nodes(void)
{
    std::string line;
    std::getline(stream, line);
    auto nnode = std::stoul(line);

    m_nds.remake(small_vector<size_t>{nnode, 3}, 0);

    while (std::getline(stream, line) && line.find('$') == std::string::npos)
    {
        auto tokens = tokenize(line, ' ');
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

void Gmsh::load_elements(void)
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
        auto tokens = tokenize(line, ' ');

        // parse element type
        auto tpn = std::stoul(tokens[1]);
        auto eldef = GmshElementDef::by_id(tpn);
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

} /* end namespace inout */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
