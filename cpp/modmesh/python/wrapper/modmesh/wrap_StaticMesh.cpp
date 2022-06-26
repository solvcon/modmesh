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

#include <modmesh/python/wrapper/modmesh/modmesh.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapStaticMesh
    : public WrapBase<WrapStaticMesh, StaticMesh, std::shared_ptr<StaticMesh>>
{

public:

    using base_type = WrapBase<WrapStaticMesh, StaticMesh, std::shared_ptr<StaticMesh>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapStaticMesh(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapStaticMesh */

WrapStaticMesh::WrapStaticMesh(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    using uint_type = typename wrapped_type::uint_type;

    (*this)
        .def_timed(
            py::init(
                [](uint8_t ndim, uint_type nnode, uint_type nface, uint_type ncell)
                { return wrapped_type::construct(ndim, nnode, nface, ncell); }),
            py::arg("ndim"),
            py::arg("nnode"),
            py::arg("nface") = 0,
            py::arg("ncell") = 0)
        //
        ;

#define MM_DECL_STATIC(NAME) \
    .def_property_readonly_static(#NAME, [](py::object const &) { return wrapped_type::NAME; })

    // clang-format off
        (*this)
            MM_DECL_STATIC(FCMND)
            MM_DECL_STATIC(CLMND)
            MM_DECL_STATIC(CLMFC)
            MM_DECL_STATIC(FCREL)
            MM_DECL_STATIC(BFREL)
        ;
    // clang-format on

#undef MM_DECL_STATIC

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("nnode", &wrapped_type::nnode)
        .def_property_readonly("nface", &wrapped_type::nface)
        .def_property_readonly("ncell", &wrapped_type::ncell)
        .def_property_readonly("nbound", &wrapped_type::nbound)
        .def_property_readonly("ngstnode", &wrapped_type::ngstnode)
        .def_property_readonly("ngstface", &wrapped_type::ngstface)
        .def_property_readonly("ngstcell", &wrapped_type::ngstcell)
        .def_property_readonly("nedge", &wrapped_type::nedge)
        .def_property_readonly("nbcs", &wrapped_type::nbcs);

    (*this)
        .def_timed("build_interior", &wrapped_type::build_interior, py::arg("_do_metric") = true, py::arg("_build_edge") = true)
        .def_timed("build_boundary", &wrapped_type::build_boundary)
        .def_timed("build_ghost", &wrapped_type::build_ghost)
        .def_timed("build_edge", &wrapped_type::build_edge);

#define MM_DECL_ARRAY(NAME) \
    .expose_SimpleArray(#NAME, [](wrapped_type & self) -> decltype(auto) { return self.NAME(); })

    // clang-format off
        (*this)
            MM_DECL_ARRAY(ndcrd)
            MM_DECL_ARRAY(fccnd)
            MM_DECL_ARRAY(fcnml)
            MM_DECL_ARRAY(fcara)
            MM_DECL_ARRAY(clcnd)
            MM_DECL_ARRAY(clvol)
            MM_DECL_ARRAY(fctpn)
            MM_DECL_ARRAY(cltpn)
            MM_DECL_ARRAY(clgrp)
            MM_DECL_ARRAY(fcnds)
            MM_DECL_ARRAY(fccls)
            MM_DECL_ARRAY(clnds)
            MM_DECL_ARRAY(clfcs)
            MM_DECL_ARRAY(ednds)
            MM_DECL_ARRAY(bndfcs)
        ;
    // clang-format on

#undef MM_DECL_ARRAY

    this->cls().attr("NONCELLTYPE") = uint8_t(CellType::NONCELLTYPE);
    this->cls().attr("POINT") = uint8_t(CellType::POINT);
    this->cls().attr("LINE") = uint8_t(CellType::LINE);
    this->cls().attr("QUADRILATERAL") = uint8_t(CellType::QUADRILATERAL);
    this->cls().attr("TRIANGLE") = uint8_t(CellType::TRIANGLE);
    this->cls().attr("HEXAHEDRON") = uint8_t(CellType::HEXAHEDRON);
    this->cls().attr("TETRAHEDRON") = uint8_t(CellType::TETRAHEDRON);
    this->cls().attr("PRISM") = uint8_t(CellType::PRISM);
    this->cls().attr("PYRAMID") = uint8_t(CellType::PYRAMID);
}

void wrap_StaticMesh(pybind11::module & mod)
{
    WrapStaticMesh::commit(mod, "StaticMesh", "StaticMesh");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
