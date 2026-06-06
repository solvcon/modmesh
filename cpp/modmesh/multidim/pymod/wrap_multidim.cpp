/*
 * Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/multidim/multidim.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapGradientElement
    : public WrapBase<WrapGradientElement, GradientElement>
{

public:

    using base_type = WrapBase<WrapGradientElement, GradientElement>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapGradientElement(pybind11::module & mod, const char * pyname, const char * clsdoc);

    static void check_ifl(wrapped_type const & self, wrapped_type::int_type ifl);
    static void check_d(wrapped_type const & self, wrapped_type::int_type d);

}; /* end class WrapGradientElement */

void WrapGradientElement::check_ifl(wrapped_type const & self, wrapped_type::int_type ifl)
{
    if (ifl < 0 || ifl >= self.clnfc())
    {
        throw std::out_of_range(std::format(
            "GradientElement: ifl {} out of range [0, {})", ifl, self.clnfc()));
    }
}

void WrapGradientElement::check_d(wrapped_type const & self, wrapped_type::int_type d)
{
    if (d < 0 || d >= self.ndim())
    {
        throw std::out_of_range(std::format(
            "GradientElement: d {} out of range [0, {})", d, self.ndim()));
    }
}

WrapGradientElement::WrapGradientElement(pybind11::module & mod, const char * pyname, const char * clsdoc)
    : base_type(mod, pyname, clsdoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](std::shared_ptr<StaticMesh> const & mesh,
                   SimpleArray<wrapped_type::real_type> const & cecnd,
                   wrapped_type::int_type icl,
                   wrapped_type::real_type tau)
                {
                    return wrapped_type(*mesh, cecnd, icl, tau);
                }),
            py::arg("mesh"),
            py::arg("cecnd"),
            py::arg("icl"),
            py::arg("tau"),
            py::keep_alive<1, 2>())
        .def_property_readonly("icl", &wrapped_type::icl)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("clnfc", &wrapped_type::clnfc)
        .def(
            "rcl",
            [](wrapped_type const & self, wrapped_type::int_type ifl)
            {
                check_ifl(self, ifl);
                return self.rcl(ifl);
            },
            py::arg("ifl"))
        .def(
            "idis",
            [](wrapped_type const & self, wrapped_type::int_type ifl, wrapped_type::int_type d)
            {
                check_ifl(self, ifl);
                check_d(self, d);
                return self.idis(ifl, d);
            },
            py::arg("ifl"),
            py::arg("d"))
        .def(
            "jdis",
            [](wrapped_type const & self, wrapped_type::int_type ifl, wrapped_type::int_type d)
            {
                check_ifl(self, ifl);
                check_d(self, d);
                return self.jdis(ifl, d);
            },
            py::arg("ifl"),
            py::arg("d"))
        //
        ;
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapEulerCore
    : public WrapBase<WrapEulerCore, EulerCore, std::shared_ptr<EulerCore>>
{

public:

    using base_type = WrapBase<WrapEulerCore, EulerCore, std::shared_ptr<EulerCore>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapEulerCore(pybind11::module & mod, const char * pyname, const char * clsdoc);

    WrapEulerCore & wrap_management();
    WrapEulerCore & wrap_preparation();
    WrapEulerCore & wrap_array();

}; /* end class WrapEulerCore */

WrapEulerCore::WrapEulerCore(pybind11::module & mod, const char * pyname, const char * clsdoc)
    : base_type(mod, pyname, clsdoc)
{
    (*this)
        .wrap_management()
        .wrap_preparation()
        .wrap_array()
        //
        ;
}

WrapEulerCore & WrapEulerCore::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](std::shared_ptr<StaticMesh> const & mesh, wrapped_type::real_type time_increment)
                {
                    return wrapped_type::construct(mesh, time_increment);
                }),
            py::arg("mesh"),
            py::arg("time_increment"))
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("ncell", &wrapped_type::ncell)
        .def_property_readonly("ngstcell", &wrapped_type::ngstcell)
        .def_property_readonly("time_increment", &wrapped_type::time_increment)
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_preparation()
{
    (*this)
        .def_timed("prepare_ce", &wrapped_type::prepare_ce)
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_array()
{
#define MM_DECL_ARRAY(NAME) \
    .expose_SimpleArray(#NAME, [](wrapped_type & self) -> decltype(auto) { return self.NAME(); })

    // clang-format off
    (*this)
        MM_DECL_ARRAY(cevol)
        MM_DECL_ARRAY(cecnd)
        MM_DECL_ARRAY(sfcnd)
        MM_DECL_ARRAY(sfnml)
        //
        ;
    // clang-format on

#undef MM_DECL_ARRAY

    return *this;
}

void wrap_multidim(pybind11::module & mod)
{
    WrapGradientElement::commit(mod, "GradientElement", "Gradient element for a single cell");
    WrapEulerCore::commit(mod, "EulerCore", "Solve the Euler equation");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
