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
                    [](const py::bytes & data)
                    { return std::make_shared<inout::Plot3d>(data); }),
                py::arg("data"))
            .def("to_block", &wrapped_type::to_block);
        ;
    }

}; /* end class WrapPlot3d */

void wrap_Plot3d(pybind11::module & mod)
{
    WrapPlot3d::commit(mod, "Plot3d", "Plot3d");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
