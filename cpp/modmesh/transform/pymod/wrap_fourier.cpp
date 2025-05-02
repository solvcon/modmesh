/*
 * Copyright (c) 2025, Chun-Hsu Lai <as2266317@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
