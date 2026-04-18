/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/python/common.hpp> // Must be the first include.

#include <modmesh/onedim/onedim.hpp>

namespace modmesh
{

namespace python
{

using namespace modmesh::onedim; // NOLINT(google-build-using-namespace)

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapEuler1DCore
    : public WrapBase<WrapEuler1DCore, Euler1DCore, std::shared_ptr<Euler1DCore>>
{

public:

    using base_type = WrapBase<WrapEuler1DCore, Euler1DCore, std::shared_ptr<Euler1DCore>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapEuler1DCore(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {

        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](size_t ncoord, double time_increment)
                    {
                        return wrapped_type::construct(ncoord, time_increment);
                    }),
                py::arg("ncoord"),
                py::arg("time_increment"))
            .def("__str__", &detail::to_str<wrapped_type>)
            .def_timed("clone", &wrapped_type::clone)
            .def_property_readonly_static(
                "nvar",
                [](py::handle const &)
                { return size_t(wrapped_type::NVAR); })
            .def_property_readonly("time_increment", &wrapped_type::time_increment)
            .def_property_readonly("ncoord", &wrapped_type::ncoord);

        (*this)
            .def_property_readonly(
                "density",
                [](wrapped_type & self)
                { return to_ndarray(self.density()); })
            .def_property_readonly(
                "velocity",
                [](wrapped_type & self)
                { return to_ndarray(self.velocity()); })
            .def_property_readonly(
                "pressure",
                [](wrapped_type & self)
                { return to_ndarray(self.pressure()); })
            .def_property_readonly(
                "temperature",
                [](wrapped_type & self)
                { return to_ndarray(self.temperature()); })
            .def_property_readonly(
                "internal_energy",
                [](wrapped_type & self)
                { return to_ndarray(self.internal_energy()); })
            .def_property_readonly(
                "entropy",
                [](wrapped_type & self)
                { return to_ndarray(self.entropy()); });

        (*this)
            .def_property_readonly(
                "gamma",
                [](wrapped_type & self)
                { return to_ndarray(self.gamma()); });

        (*this)
            .def_property_readonly(
                "coord",
                [](wrapped_type & self)
                { return to_ndarray(self.coord()); })
            .def_property_readonly(
                "cfl",
                [](wrapped_type & self)
                { return to_ndarray(self.cfl()); })
            .def_property_readonly(
                "so0",
                [](wrapped_type & self)
                { return to_ndarray(self.so0()); })
            .def_property_readonly(
                "so1",
                [](wrapped_type & self)
                { return to_ndarray(self.so1()); });

        (*this)
            .def_timed("update_cfl", &wrapped_type::update_cfl, py::arg("odd_plane"))
            .def_timed("march_half_so0", &wrapped_type::march_half_so0, py::arg("odd_plane"))
            .def_timed("treat_boundary_so0", &wrapped_type::treat_boundary_so0)
            .def_timed("treat_boundary_so1", &wrapped_type::treat_boundary_so1)
            .def_timed("setup_march", &wrapped_type::setup_march);

        (*this)
            .def_group_so1<1>()
            .def_group_so1<2>();
    }

    template <size_t ALPHA>
    wrapper_type & def_group_so1()
    {
        // NOLINTNEXTLINE(misc-unused-alias-decls)
        namespace py = pybind11;

        (*this)
            .def_timed(
                std::format("march_half_so1_alpha{}", ALPHA).c_str(),
                [](wrapped_type & self, bool odd_plane)
                {
                    return self.template march_half_so1_alpha<ALPHA>(odd_plane);
                },
                py::arg("odd_plane"))
            .def_timed(
                std::format("march_half1_alpha{}", ALPHA).c_str(),
                [](wrapped_type & self)
                {
                    self.template march_half1_alpha<ALPHA>();
                })
            .def_timed(
                std::format("march_half2_alpha{}", ALPHA).c_str(),
                [](wrapped_type & self)
                {
                    self.template march_half2_alpha<ALPHA>();
                })
            .def_timed(
                std::format("march_alpha{}", ALPHA).c_str(),
                [](wrapped_type & self, size_t steps)
                {
                    self.template march_alpha<ALPHA>(steps);
                },
                py::arg("steps"));

        return *this;
    }

}; /* end class WrapEuler1DCore */

void wrap_onedim(pybind11::module & mod)
{
    mod.doc() = "One-dimensional space-time CESE method code";

    WrapEuler1DCore::commit(mod, "Euler1DCore", "Solve the Euler equation");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
