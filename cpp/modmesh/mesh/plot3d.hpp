#include <modmesh/mesh/StaticMesh.hpp>
#include <fstream>

namespace modmesh
{
namespace inout
{

class Plot3d
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Plot3d(const std::string & filepath);

    ~Plot3d() = default;

    Plot3d() = delete;
    Plot3d(Plot3d const & other) = delete;
    Plot3d(Plot3d && other) = delete;
    Plot3d & operator=(Plot3d const & other) = delete;
    Plot3d & operator=(Plot3d && other) = delete;

    uint8_t ndim = 3;
    uint_type nnode = 0;
    uint_type nface = 6;
    uint_type ncell = 0;
    int Imax = 0, Jmax = 0, Kmax = 0;

    uint8_t get_ndim() const { return ndim; }
    uint_type get_nnode() const { return nnode; }
    uint_type get_nface() const { return nface; }
    uint_type get_ncell() const { return ncell; }

    void load_file(const std::string & filepath);

    void paser(const std::string & filepath)
    {

        load_file(filepath);
    }

private:

    void ParsePlot3dFile(std::ifstream & p3d_File)
    {
        //  Parse the 3d file and populate the StaticMesh
        //  This will involve reading the coordinates and other data
        //  from the file and using StaticMesh methods and properties
        //  to store this data.

        auto mesh = StaticMesh::construct(ndim, nnode, nface, ncell);
        (*mesh).set_ndim(static_cast<int>(3));

        p3d_File >> ncell;
        (*mesh).set_ncell(ncell);

        for (size_t i = 0; i < (*mesh).ncell(); i++)
        {
            p3d_File >> Imax >> Jmax >> Kmax;
            nnode += Imax * Jmax * Kmax;
        }

        (*mesh).set_nnode(nnode);
        (*mesh).set_nface(6);

        mesh->ndcrd().remake(small_vector<size_t>{(*mesh).nnode(), 3}, 0);

        // Reading coordinates and populating 'ndcrd'
        double x, y, z;
        for (size_t i = 0; i < (*mesh).nnode(); ++i)
        {
            p3d_File >> x;
            mesh->ndcrd(i, 0) = x;
        }
        for (size_t i = 0; i < (*mesh).nnode(); ++i)
        {
            p3d_File >> y;
            mesh->ndcrd(i, 1) = y;
        }
        for (size_t i = 0; i < (*mesh).nnode(); ++i)
        {
            p3d_File >> z;
            mesh->ndcrd(i, 2) = z;
        }
        // Reading connectivity and populating 'clnds'
        mesh->createHexFaces();
        p3d_File.close();
    }

}; /* end class Plot3d */

} /* end namespace inout */
} /* end namespace modmesh */