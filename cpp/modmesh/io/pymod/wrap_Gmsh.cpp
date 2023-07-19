#include <modmesh/io/pymod/io_pymod.hpp>
#include <modmesh/modmesh.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapGmsh
    : public WrapBase<WrapGmsh, IO::Gmsh::Gmsh, std::shared_ptr<IO::Gmsh::Gmsh>>
{
public:

    using base_type = WrapBase<WrapGmsh, IO::Gmsh::Gmsh, std::shared_ptr<IO::Gmsh::Gmsh>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapGmsh(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapGmsh, IO::Gmsh::Gmsh, std::shared_ptr<IO::Gmsh::Gmsh>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                py::init(
                    [](const std::string & path)
                    { return std::make_shared<IO::Gmsh::Gmsh>(path); }),
                py::arg("file_name"))
            .def("toblock", &wrapped_type::toblock);
        ;
    }

}; /* end class WrapGmsh */

void wrap_Gmsh(pybind11::module & mod)
{
    WrapGmsh::commit(mod, "Gmsh", "Gmsh");
}

} /* end namespace python */

} /* end namespace modmesh */
