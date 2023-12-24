#include <modmesh/mesh/StaticMesh.hpp>
#include <fstream>

namespace modmesh
{
namespace inout
{
struct Plot3dElementDef
{
    static Plot3dElementDef by_id(uint16_t id);

    Plot3dElementDef(uint8_t ndim, uint16_t nnds, uint8_t mmtpn, small_vector<uint8_t> const & mmcl)
        : m_ndim(ndim)
        , m_nnds(nnds)
        , m_mmtpn(mmtpn)
        , m_mmcl(mmcl)
    {
    }

    ~Plot3dElementDef() = default;

    Plot3dElementDef() = default;
    Plot3dElementDef(const Plot3dElementDef & other) = delete;
    Plot3dElementDef(Plot3dElementDef && other) = delete;
    Plot3dElementDef & operator=(const Plot3dElementDef & other) = delete;
    Plot3dElementDef & operator=(Plot3dElementDef && other) = delete;

    uint8_t ndim() const { return m_ndim; }
    uint16_t nnds() const { return m_nnds; }
    uint8_t mmtpn() const { return m_mmtpn; }
    small_vector<uint8_t> mmcl() const { return m_mmcl; }

private:
    uint8_t m_ndim = 0; /* Number of dimension */ // me is 3
    uint16_t m_nnds = 0; /* Number of nodes     */ // me is 8
    uint8_t m_mmtpn = 0; /* modmesh cell type   */ // me is 5
    small_vector<uint8_t> m_mmcl; /* modmesh cell order  */ // guess:mh.clnds.ndarray[:, :9] = [(8, 0 ,2 ,3 ,1 ,4 ,6 ,7 ,5)]
}; /* end struct Plot3dElementDef */

class Plot3d
    : public NumberBase<int32_t, double>
{

public:
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:

    explicit Plot3d();
    ~Plot3d() = default;

    Plot3d(Plot3d const & other) = delete;
    Plot3d(Plot3d && other) = delete;
    Plot3d & operator=(Plot3d const & other) = delete;
    Plot3d & operator=(Plot3d && other) = delete;

    std::shared_ptr<StaticMesh> load_file(const std::string & filepath);
    std::shared_ptr<StaticMesh> ParsePlot3dFile(std::ifstream & p3d_File);

    uint8_t ndim = 3;
    uint_type nnode = 0;
    uint_type nface = 6;
    uint_type ncell = 0;
    size_t nblock = 0;
    size_t Imax = 0, Jmax = 0, Kmax = 0;

    uint8_t get_ndim() const { return ndim; }
    uint_type get_nnode() const { return nnode; }
    uint_type get_nface() const { return nface; }
    uint_type get_ncell() const { return ncell; }
    uint_type get_nblock() const { return nblock; }

}; /* end class Plot3d */

} /* end namespace inout */
} /* end namespace modmesh */
