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

#include <modmesh/mesh/pymod/mesh_pymod.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

namespace modmesh
{

namespace python
{

template <typename Wrapper, typename GT>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapStaticGridBase
    : public WrapBase<Wrapper, GT>
{

public:

    using base_type = WrapBase<Wrapper, GT>;
    using wrapped_type = typename base_type::wrapped_type;

    using serial_type = typename wrapped_type::serial_type;
    using real_type = typename wrapped_type::real_type;

    friend typename base_type::root_base_type;

protected:

    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    WrapStaticGridBase(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_property_readonly_static(
                "NDIM",
                [](py::object const &)
                { return wrapped_type::NDIM; })
            //
            ;
    }

}; /* end class WrapStaticGridBase */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapStaticGrid1d
    : public WrapStaticGridBase<WrapStaticGrid1d, StaticGrid1d>
{

public:

    friend root_base_type;

    using base_type = WrapStaticGridBase<WrapStaticGrid1d, StaticGrid1d>;

protected:

    WrapStaticGrid1d(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapStaticGrid1d */

WrapStaticGrid1d::WrapStaticGrid1d(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](serial_type nx)
                { return std::make_unique<StaticGrid1d>(nx); }),
            py::arg("nx"))
        .def(
            "__len__",
            [](wrapped_type const & self)
            { return self.size(); })
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, wrapped_type::real_type val)
            { self.at(it) = val; })
        .def_property_readonly(
            "nx",
            [](wrapped_type const & self)
            { return self.nx(); })
        .expose_SimpleArray(
            "coord",
            [](wrapped_type & self) -> decltype(auto)
            { return self.coord(); })
        .def_timed("fill", &wrapped_type::fill, py::arg("value"))
        //
        ;
}

// clang-format off
#define MM_DECL_StaticGridMD(NDIM) \
class \
MODMESH_PYTHON_WRAPPER_VISIBILITY \
WrapStaticGrid ## NDIM ## d \
  : public WrapStaticGridBase< WrapStaticGrid ## NDIM ## d, StaticGrid ## NDIM ## d > \
{ \
\
public: \
\
      friend root_base_type; \
\
    using base_type = WrapStaticGridBase< WrapStaticGrid ## NDIM ## d, StaticGrid ## NDIM ## d >; \
\
protected: \
\
    explicit WrapStaticGrid ## NDIM ## d(pybind11::module & mod, char const * pyname, char const * pydoc) \
      : base_type(mod, pyname, pydoc) \
    {} \
\
}
// clang-format on

MM_DECL_StaticGridMD(2);
MM_DECL_StaticGridMD(3);

#undef MM_DECL_StaticGridMD

void wrap_StaticGrid(pybind11::module & mod)
{
    WrapStaticGrid1d::commit(mod, "StaticGrid1d", "StaticGrid1d");
    WrapStaticGrid2d::commit(mod, "StaticGrid2d", "StaticGrid2d");
    WrapStaticGrid3d::commit(mod, "StaticGrid3d", "StaticGrid3d");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
