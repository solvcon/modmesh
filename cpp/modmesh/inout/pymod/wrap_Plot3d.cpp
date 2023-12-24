#include <modmesh/inout/pymod/inout_pymod.hpp>
#include <modmesh/modmesh.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPlot3d
    : public WrapBase<WrapPlot3d, inout::Plot3d, std::shared_ptr<inout::Plot3d>>
{
public:

    using base_type = WrapBase<WrapPlot3d, inout::Plot3d, std::shared_ptr<inout::Plot3d>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapPlot3d(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapPlot3d, inout::Plot3d, std::shared_ptr<inout::Plot3d>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(py::init([]()
                          { return std::make_shared<inout::Plot3d>(); }),
                 "Constructor without file loading")
            .def("load_file", &wrapped_type::load_file, py::arg("filepath"));

        (*this)
            .def_property_readonly("ndim", &wrapped_type::get_ndim)
            .def_property_readonly("nnode", &wrapped_type::get_nnode)
            .def_property_readonly("nface", &wrapped_type::get_nface)
            .def_property_readonly("ncell", &wrapped_type::get_ncell);
    }

}; /* end class WrapPlot3d */

void wrap_Plot3d(pybind11::module & mod)
{
    WrapPlot3d::commit(mod, "Plot3d", "Plot3d");
}

} /* end namespace python */

} /* end namespace modmesh */
