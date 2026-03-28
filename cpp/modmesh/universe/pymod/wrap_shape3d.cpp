/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/universe/pymod/universe_pymod.hpp> // Must be the first include.

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapBoundBox3d
    : public WrapBase<WrapBoundBox3d<T>, BoundBox3d<T>>
{
public:
    using base_type = WrapBase<WrapBoundBox3d<T>, BoundBox3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = T;

    friend typename base_type::root_base_type;

protected:
    WrapBoundBox3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapBoundBox3d & wrap_management();
    WrapBoundBox3d & wrap_geometry();
}; /* end class WrapBoundBox3d */

template <typename T>
WrapBoundBox3d<T>::WrapBoundBox3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapBoundBox3d<T> & WrapBoundBox3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<value_type, value_type, value_type, value_type, value_type, value_type>(),
             py::arg("min_x"),
             py::arg("min_y"),
             py::arg("min_z"),
             py::arg("max_x"),
             py::arg("max_y"),
             py::arg("max_z"))
        //
        ;

    return *this;
}

template <typename T>
WrapBoundBox3d<T> & WrapBoundBox3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    // Bounding value.
    (*this)
        .def_property_readonly("min_x", &wrapped_type::min_x)
        .def_property_readonly("min_y", &wrapped_type::min_y)
        .def_property_readonly("min_z", &wrapped_type::min_z)
        .def_property_readonly("max_x", &wrapped_type::max_x)
        .def_property_readonly("max_y", &wrapped_type::max_y)
        .def_property_readonly("max_z", &wrapped_type::max_z)
        //
        ;

    // Calculation.
    (*this)
        .def("calc_area", &wrapped_type::calc_area)
        .def("overlap", &wrapped_type::overlap, py::arg("other"))
        .def("contain", &wrapped_type::contain, py::arg("other"))
        .def("expand", &wrapped_type::expand, py::arg("other"))
        //
        ;

    return *this;
}

void wrap_shape3d(pybind11::module & mod)
{
    WrapBoundBox3d<float>::commit(mod, "BoundBox3dFp32", "BoundBox3dFp32");
    WrapBoundBox3d<double>::commit(mod, "BoundBox3dFp64", "BoundBox3dFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: