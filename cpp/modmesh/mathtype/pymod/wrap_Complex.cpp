#include <pybind11/operators.h>
#include <modmesh/modmesh.hpp>
#include <modmesh/mathtype/pymod/mathtype_pymod.hpp>

namespace modmesh
{

namespace python
{
template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapComplex
    : public WrapBase<WrapComplex<T>, modmesh::Complex<T>, std::shared_ptr<modmesh::Complex<T>>>
{
    using base_type = WrapBase<WrapComplex<T>, modmesh::Complex<T>, std::shared_ptr<modmesh::Complex<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = T;

    friend base_type;

    WrapComplex(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapComplex<value_type>, modmesh::Complex<value_type>, std::shared_ptr<modmesh::Complex<value_type>>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                py::init(
                    [](const value_type & real_v, const value_type & imag_v)
                    { return std::make_shared<wrapped_type>(real_v, imag_v); }),
                py::arg("real_v"),
                py::arg("imag_v"))
            .def(
                py::init(
                    []()
                    { return std::make_shared<wrapped_type>(); }))
            .def(
                py::init(
                    [](const wrapped_type & other)
                    { return std::make_shared<wrapped_type>(other); }),
                py::arg("other"))
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / value_type())
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def_property_readonly("real", &wrapped_type::real)
            .def_property_readonly("imag", &wrapped_type::imag)
            .def("norm", &wrapped_type::norm);
    }

}; /* end class WrapComplex */

void wrap_Complex(pybind11::module & mod)
{
    WrapComplex<float>::commit(mod, "ComplexFloat32", "ComplexFloat32");
    WrapComplex<double>::commit(mod, "ComplexFloat64", "ComplexFloat64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
