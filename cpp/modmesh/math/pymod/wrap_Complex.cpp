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

#include <modmesh/math/pymod/math_pymod.hpp>

#include <modmesh/modmesh.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

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
        static const auto complex_dtype = py::dtype(std::is_same<value_type, float>::value ? "complex64" : "complex128");

        PYBIND11_NUMPY_DTYPE(wrapped_type, real_v, imag_v);

        (*this)
            .def(
                py::init(
                    []()
                    { return std::make_shared<wrapped_type>(); }))
            .def(
                py::init(
                    [](const value_type & real, const value_type & imag)
                    { return std::make_shared<wrapped_type>(wrapped_type{real, imag}); }),
                py::arg("real"),
                py::arg("imag"))
            .def(
                py::init(
                    [](const wrapped_type & other)
                    { return std::make_shared<wrapped_type>(wrapped_type{other.real_v, other.imag_v}); }),
                py::arg("other"))
            .def(py::self + py::self) // NOLINT(misc-redundant-expression)
            .def(py::self + value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() + py::self) // NOLINT(misc-redundant-expression)
            .def(py::self - py::self) // NOLINT(misc-redundant-expression)
            .def(py::self - value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() - py::self) // NOLINT(misc-redundant-expression)
            .def(py::self * py::self) // NOLINT(misc-redundant-expression)
            .def(py::self * value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() * py::self) // NOLINT(misc-redundant-expression)
            .def(py::self / py::self) // NOLINT(misc-redundant-expression)
            .def(py::self / value_type()) // NOLINT(misc-redundant-expression)
            .def(value_type() / py::self) // NOLINT(misc-redundant-expression)
            .def(py::self += py::self) // NOLINT(misc-redundant-expression)
            .def(py::self += value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self -= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self -= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self *= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self *= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self /= py::self) // NOLINT(misc-redundant-expression)
            .def(py::self /= value_type()) // NOLINT(misc-redundant-expression)
            .def(py::self == py::self) // NOLINT(misc-redundant-expression)
            .def(py::self != py::self) // NOLINT(misc-redundant-expression)
            .def(py::self < py::self) // NOLINT(misc-redundant-expression)
            .def(py::self > py::self) // NOLINT(misc-redundant-expression)
            .def_readonly("real", &wrapped_type::real_v)
            .def_readonly("imag", &wrapped_type::imag_v)
            .def("norm", &wrapped_type::norm)
            .def("conj", &wrapped_type::conj)
            .def("dtype",
                 []()
                 { return complex_dtype; });
    }

}; /* end class WrapComplex */

void wrap_Complex(pybind11::module & mod)
{
    WrapComplex<float>::commit(mod, "complex64", "complex64");
    WrapComplex<double>::commit(mod, "complex128", "complex128");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
