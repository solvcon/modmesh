/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/solvcon.hpp>

#include <solvcon/transform/pymod/transform_pymod.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapFourierTransform
    : public WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>
{
    using base_type = WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapFourierTransform(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_static("fft", &wrapped_type::fft<solvcon::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("fft", &wrapped_type::fft<solvcon::Complex, float>, py::arg("input"), py::arg("output"))
            .def_static("ifft", &wrapped_type::ifft<solvcon::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("ifft", &wrapped_type::ifft<solvcon::Complex, float>, py::arg("input"), py::arg("output"))
            .def_static("dft", &wrapped_type::dft<solvcon::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("dft", &wrapped_type::dft<solvcon::Complex, float>, py::arg("input"), py::arg("output"));
    }

}; /* end class WrapFourierTransform */

void wrap_FourierTransform(pybind11::module & mod)
{
    WrapFourierTransform::commit(mod, "FourierTransform", "Fourier transform library");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
