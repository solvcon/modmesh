#include <modmesh/inout/plot3d.hpp>

namespace modmesh
{
namespace inout
{

void Plot3d::load_file(const std::string & filepath)
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
            ParsePlot3dFile(p3d_File);
        }
        // else if(extension==".p2d"){
        // Parse2DFile logics go here
        //}
        else
        {
            std::cerr << "Unsupported file extension." << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

Plot3d::Plot3d(const std::string & filepath)
{
    load_file(filepath);
}

} // namespace inout
} // namespace modmesh