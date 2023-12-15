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
            .def(
                py::init(
                    [](const std::string & filepath)
                    { return std::make_shared<inout::Plot3d>(filepath); }),
                py::arg("filepath"))
            .def("load_file", &wrapped_type::load_file);
        ;
    }

}; /* end class WrapPlot3d */

void wrap_Plot3d(pybind11::module & mod)
{
    WrapPlot3d::commit(mod, "Plot3d", "Plot3d");
}

} /* end namespace python */

} /* end namespace modmesh */
