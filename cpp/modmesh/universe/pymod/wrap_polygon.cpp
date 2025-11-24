/*
 * Copyright (c) 2025, An-Chi Liu <phy.tiger@gmail.com>
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

#include <modmesh/universe/pymod/universe_pymod.hpp>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPolygon3d
    : public WrapBase<WrapPolygon3d<T>, Polygon3d<T>, std::shared_ptr<Polygon3d<T>>>
{
public:
    using base_type = WrapBase<WrapPolygon3d<T>, Polygon3d<T>, std::shared_ptr<Polygon3d<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_pad_type = typename wrapped_type::segment_pad_type;
    using curve_pad_type = typename wrapped_type::curve_pad_type;

    friend typename base_type::root_base_type;

protected:
    WrapPolygon3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};

template <typename T>
WrapPolygon3d<T>::WrapPolygon3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](std::shared_ptr<SegmentPad<T>> segments)
                {
                    return wrapped_type::construct(segments);
                }),
            py::arg("segments"))
        .def(
            py::init(
                [](std::shared_ptr<CurvePad<T>> curves, value_type sample_length)
                {
                    return wrapped_type::construct(curves, sample_length);
                }),
            py::arg("curves"),
            py::arg("sample_length"))
        .def(
            py::init(
                [](std::shared_ptr<SegmentPad<T>> segments, std::shared_ptr<CurvePad<T>> curves, value_type sample_length)
                {
                    return wrapped_type::construct(segments, curves, sample_length);
                }),
            py::arg("segments"),
            py::arg("curves"),
            py::arg("sample_length"))
        .def_property_readonly("segments", &wrapped_type::segments)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("size", &wrapped_type::size)
        .def("get", &wrapped_type::get, py::arg("index"))
        .def("get_at", &wrapped_type::get_at, py::arg("index"))
        .def("calc_bound_box", &wrapped_type::calc_bound_box)
        .def(
            "search_segments",
            [](wrapped_type const & self, BoundBox3d<T> const & box)
            {
                std::vector<Segment3d<T>> results;
                self.search_segments(box, results);
                return results;
            },
            py::arg("box"))
        .def("rebuild_rtree", &wrapped_type::rebuild_rtree)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        ;
}

void wrap_polygon(pybind11::module & mod)
{
    WrapPolygon3d<float>::commit(mod, "Polygon3dFp32", "Polygon3dFp32");
    WrapPolygon3d<double>::commit(mod, "Polygon3dFp64", "Polygon3dFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
