/*
 * Copyright (c) 2026, Anchi Liu <phy.tiger@gmail.com>
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

#include <memory>

#include <modmesh/linalg/pymod/linalg_pymod.hpp>

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapLuFactorization
    : public WrapBase<WrapLuFactorization<T>, LuFactorization<T>>
{

    using root_base_type = WrapBase<WrapLuFactorization<T>, LuFactorization<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend root_base_type;

    WrapLuFactorization(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapLuFactorization */

template <typename T>
WrapLuFactorization<T>::WrapLuFactorization(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a)
                {
                    return std::make_unique<wrapped_type>(a);
                }),
            py::arg("a"));

    (*this)
        .def_property_readonly(
            "lu",
            &wrapped_type::lu,
            py::return_value_policy::reference_internal)
        // piv is a SimpleArray<int64_t>; the SimpleArray caster maps it to a
        // SimpleArrayInt64 that shares the cached buffer, so tie its lifetime
        // to the owner like lu above.
        .def_property_readonly(
            "piv",
            &wrapped_type::piv,
            py::return_value_policy::reference_internal)
        .def_property_readonly("n", &wrapped_type::n)
        .def(
            "solve",
            &wrapped_type::solve,
            py::arg("b"),
            "Solve A x = b using the cached LU factors.")
        .def(
            "inv",
            &wrapped_type::inv,
            "Compute A^(-1) using the cached LU factors.")
        .def(
            "det",
            [](wrapped_type const & self)
            {
                // Complex<T> has no pybind11 caster; convert to std::complex
                // (which pybind11 maps to a Python complex) for complex T.
                if constexpr (is_complex_v<T>)
                {
                    return self.det().to_std_complex();
                }
                else
                {
                    return self.det();
                }
            },
            "Compute det(A) using the cached LU factors.");
}

void wrap_LuFactorization(pybind11::module & mod)
{
    WrapLuFactorization<float>::commit(
        mod, "LuFactorizationFloat32", "LU factorization (float32)");
    WrapLuFactorization<double>::commit(
        mod, "LuFactorizationFloat64", "LU factorization (float64)");
    WrapLuFactorization<Complex<float>>::commit(
        mod, "LuFactorizationComplex64", "LU factorization (complex64)");
    WrapLuFactorization<Complex<double>>::commit(
        mod, "LuFactorizationComplex128", "LU factorization (complex128)");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
