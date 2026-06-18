/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/modmesh.hpp>

#include <modmesh/transform/pymod/transform_pymod.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapFourierTransform
    : public WrapBase<WrapFourierTransform, modmesh::FourierTransform, std::shared_ptr<modmesh::FourierTransform>>
{
    using base_type = WrapBase<WrapFourierTransform, modmesh::FourierTransform, std::shared_ptr<modmesh::FourierTransform>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapFourierTransform(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapFourierTransform, modmesh::FourierTransform, std::shared_ptr<modmesh::FourierTransform>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_static("fft", &wrapped_type::fft<modmesh::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("fft", &wrapped_type::fft<modmesh::Complex, float>, py::arg("input"), py::arg("output"))
            .def_static("ifft", &wrapped_type::ifft<modmesh::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("ifft", &wrapped_type::ifft<modmesh::Complex, float>, py::arg("input"), py::arg("output"))
            .def_static("dft", &wrapped_type::dft<modmesh::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("dft", &wrapped_type::dft<modmesh::Complex, float>, py::arg("input"), py::arg("output"));
    }

}; /* end class WrapFourierTransform */

void wrap_FourierTransform(pybind11::module & mod)
{
    WrapFourierTransform::commit(mod, "FourierTransform", "Fourier transform library");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
