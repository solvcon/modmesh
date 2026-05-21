/*
 * Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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

#ifdef __APPLE__

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapEigenSystem
    : public WrapBase<WrapEigenSystem, EigenSystem>
{

    using base_type = WrapBase<WrapEigenSystem, EigenSystem>;
    using wrapped_type = typename base_type::wrapped_type;
    using array_type = SimpleArray<double>;

    friend base_type;

    WrapEigenSystem(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapEigenSystem */

WrapEigenSystem::WrapEigenSystem(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a, bool do_vl, bool do_vr)
                {
                    return std::make_unique<wrapped_type>(a, do_vl, do_vr);
                }),
            py::arg("a"),
            py::arg("do_vl") = true,
            py::arg("do_vr") = true,
            // Keep the input array's Python wrapper alive while the
            // EigenSystem lives, so m_matrix's C++ reference stays
            // valid for the lifetime of this instance.
            py::keep_alive<1, 2>());

    (*this)
        .def("run", &wrapped_type::run)
        .def_property_readonly(
            "matrix",
            &wrapped_type::matrix,
            pybind11::return_value_policy::reference_internal)
        .def_property_readonly("wr", &wrapped_type::wr)
        .def_property_readonly("wi", &wrapped_type::wi)
        .def_property_readonly(
            "vl",
            [](wrapped_type const & self) -> array_type const &
            {
                return self.vl();
            })
        .def_property_readonly(
            "vr",
            [](wrapped_type const & self) -> array_type const &
            {
                return self.vr();
            })
        .def(
            "get_vl",
            &wrapped_type::vl,
            py::arg("suppress_exception") = false,
            "Left eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vl=False.")
        .def(
            "get_vr",
            &wrapped_type::vr,
            py::arg("suppress_exception") = false,
            "Right eigenvectors; suppress_exception=True returns empty "
            "instead of raising when do_vr=False.")
        .def_property_readonly("do_vl", &wrapped_type::do_vl)
        .def_property_readonly("do_vr", &wrapped_type::do_vr)
        .def_property_readonly("done", &wrapped_type::done);
}

#endif /* __APPLE__ */

void wrap_EigenSystem(pybind11::module & mod)
{
#ifdef __APPLE__
    WrapEigenSystem::commit(mod, "EigenSystem", "Eigen problem solver");
#else // __APPLE__
    mod.attr("EigenSystem") = pybind11::none();
#endif // __APPLE__
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
