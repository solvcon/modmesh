#include "StaticMesh.hpp" // Include modmesh headers
#include <fstream>

namespace modmesh
{

// The parser logic goes here
void ParsePlot3DFile(const std::string & filename)
{
    // Open file, handle errors
    std::ifstream p3d_File(filename);
    if (!p3d_File.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Parse the file and populate the StaticMesh
    // This will involve reading the coordinates and other data
    // from the file and using StaticMesh methods and properties
    // to store this data.

    uint8_t ndim = 3;
    modmesh::uint_type nnode = 0;
    modmesh::uint_type nface = 0;
    modmesh::uint_type ncell = 0;

    auto mesh = modmesh::StaticMesh::construct(ndim, nnode, nface, ncell);
    (*mesh).set_ndim(static_cast<int>(3));
    std::cout << "Dim is " << static_cast<int>((*mesh).ndim()) << std::endl;

    p3d_File >> ncell;
    (*mesh).set_ncell(ncell);
    std::cout << "ncell is: " << (*mesh).ncell() << std::endl;

    int Imax, Jmax, Kmax;

    for (size_t i = 0; i < (*mesh).ncell(); i++)
    {
        p3d_File >> Imax >> Jmax >> Kmax;
        std::cout << "Imax: " << Imax << ", Jmax: " << Jmax << ", Kmax: " << Kmax << std::endl;
        nnode += Imax * Jmax * Kmax;
    }

    (*mesh).set_nnode(nnode);
    std::cout << "nnode is: " << (*mesh).nnode() << std::endl;
    mesh->ndcrd().remake(modmesh::small_vector<size_t>{(*mesh).nnode(), 3}, 0);

    // Reading coordinates and populating 'ndcrd'
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        double x;
        p3d_File >> x;
        mesh->ndcrd(i, 0) = x;
    }
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        double y;
        p3d_File >> y;
        mesh->ndcrd(i, 1) = y;
    }
    for (size_t i = 0; i < (*mesh).nnode(); ++i)
    {
        double z;
        p3d_File >> z;
        mesh->ndcrd(i, 2) = z;
        std::cout << "ndcrd(" << i << ",2): " << mesh->ndcrd(i, 2) << std::endl;
    }

    p3d_File.close();
}

} /* end namespace modmesh */

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <Plot3D file>" << std::endl;
        return 1;
    }

    try
    {
        std::string filename = argv[1];
        std::string extension = filename.substr(filename.size() - 4);

        if (extension == ".p3d")
        {
            modmesh::ParsePlot3DFile(argv[1]);
        }
        // else if(extension==".p2d"){
        //     modmesh::ParsePlot2DFile(argv[1]);
        // }
        else
        {
            std::cerr << "Unsupported file extension." << std::endl;
            return 1;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
