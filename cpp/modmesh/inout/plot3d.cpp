#include <modmesh/inout/plot3d.hpp>

namespace modmesh
{
namespace inout
{

std::shared_ptr<StaticMesh> Plot3d::load_file(const std::string & filepath)
{
    std::ifstream p3d_File(filepath);
    if (!p3d_File.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    try
    {
        std::string extension = filepath.substr(filepath.size() - 4);

        if (extension == ".p3d")
        {
            return ParsePlot3dFile(p3d_File);
        }
        // else if(extension==".p2d"){
        // ParsePlot2DFile logics go here
        //}
        else
        {
            std::cerr << "!!Unsupported file extension." << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return std::shared_ptr<modmesh::StaticMesh>(); // return an empty std::shared_ptr if an invalid condition is detected or an exception occurs
}

std::shared_ptr<StaticMesh> Plot3d::ParsePlot3dFile(std::ifstream & p3d_File)
{

    p3d_File >> nblock;
    std::cout << "nblock: " << nblock << std::endl;
    for (size_t i = 0; i < nblock; i++)
    {
        p3d_File >> Imax >> Jmax >> Kmax;
        nnode += Imax * Jmax * Kmax;
    }
    ndim = 3;
    ncell = 6;

    std::cout << "nnode1: " << nnode << std::endl;

    std::shared_ptr<StaticMesh> mesh = StaticMesh::construct(ndim, nnode, nface, ncell);
    std::cout << "ndim: " << unsigned((*mesh).ndim()) << std::endl;

    std::cout << "nnode2: " << (*mesh).nnode() << std::endl;
    std::cout << "nface: " << (*mesh).nface() << std::endl;

    mesh->ndcrd().remake(small_vector<size_t>{(*mesh).nnode(), 3}, 0);

    // Reading coordinates and populating 'ndcrd'
    size_t x = 0, y = 0, z = 0;
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        p3d_File >> x;
        mesh->ndcrd(i, 0) = x;
        std::cout << "ndcrd(" << i << ",0): " << mesh->ndcrd(i, 0) << std::endl;
    }
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        p3d_File >> y;
        mesh->ndcrd(i, 1) = y;
        std::cout << "ndcrd(" << i << ",1): " << mesh->ndcrd(i, 1) << std::endl;
    }
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        p3d_File >> z;
        mesh->ndcrd(i, 2) = z;
        std::cout << "ndcrd(" << i << ",2): " << mesh->ndcrd(i, 2) << std::endl;
    }

    mesh->cltpn().remake(small_vector<size_t>{(*mesh).ncell() + 1}, 5);

    mesh->clnds().remake(small_vector<size_t>{7, 9}, 0);
    // Set clnds for each cell
    for (size_t i = 0; i < mesh->ncell() + 1; ++i)
    {
        mesh->clnds(i, 0) = 8;
        mesh->clnds(i, 1) = 0;
        mesh->clnds(i, 2) = 2;
        mesh->clnds(i, 3) = 3;
        mesh->clnds(i, 4) = 1;
        mesh->clnds(i, 5) = 4;
        mesh->clnds(i, 6) = 6;
        mesh->clnds(i, 7) = 7;
        mesh->clnds(i, 8) = 5;
    }
    mesh->clfcs().remake(small_vector<size_t>{(*mesh).ncell() + 1, 7}, 0);
    mesh->call_build_faces_from_cells();
    mesh->call_build_edge();
    p3d_File.close();
    return mesh;
}

Plot3d::Plot3d()
{
}

} // namespace inout
} // namespace modmesh