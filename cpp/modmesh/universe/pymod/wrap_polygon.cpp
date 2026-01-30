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
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPolygon
    : public WrapBase<WrapPolygon<T>, Polygon3d<T>>
{
public:
    using base_type = WrapBase<WrapPolygon<T>, Polygon3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_type = typename wrapped_type::segment_type;
    using polygon_pad_type = typename wrapped_type::polygon_pad_type;

    friend typename base_type::root_base_type;

protected:
    WrapPolygon(pybind11::module & mod, char const * pyname, char const * pydoc);
};

template <typename T>
WrapPolygon<T>::WrapPolygon(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_property_readonly("polygon_id", &wrapped_type::polygon_id)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("nnode", &wrapped_type::nnode)
        .def("get_node", &wrapped_type::node, py::arg("index"))
        .def("get_edge", &wrapped_type::edge, py::arg("index"))
        .def("compute_signed_area", &wrapped_type::compute_signed_area)
        .def("is_counter_clockwise", &wrapped_type::is_counter_clockwise)
        .def("calc_bound_box", &wrapped_type::calc_bound_box)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        .def("is_same", &wrapped_type::is, py::arg("other"))
        .def("is_not_same", &wrapped_type::is_not, py::arg("other"));
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPolygonPad
    : public WrapBase<WrapPolygonPad<T>, PolygonPad<T>, std::shared_ptr<PolygonPad<T>>>
{
public:
    using base_type = WrapBase<WrapPolygonPad<T>, PolygonPad<T>, std::shared_ptr<PolygonPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;
    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_pad_type = typename wrapped_type::segment_pad_type;
    using curve_pad_type = typename wrapped_type::curve_pad_type;
    using polygon_type = typename wrapped_type::polygon_type;

    friend typename base_type::root_base_type;

protected:
    WrapPolygonPad(pybind11::module & mod, char const * pyname, char const * pydoc);
};

template <typename T>
WrapPolygonPad<T>::WrapPolygonPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](uint8_t ndim)
                {
                    return wrapped_type::construct(ndim);
                }),
            py::arg("ndim"))
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("num_polygons", &wrapped_type::num_polygons)
        .def_property_readonly("num_points", &wrapped_type::num_points)
        .def(
            "add_polygon",
            [](wrapped_type & self, std::vector<point_type> const & nodes)
            {
                return self.add_polygon(nodes);
            },
            py::arg("nodes"))
        .def(
            "add_polygon_from_segments",
            &wrapped_type::add_polygon_from_segments,
            py::arg("segments"))
        .def(
            "add_polygon_from_curves",
            &wrapped_type::add_polygon_from_curves,
            py::arg("curves"),
            py::arg("sample_length"))
        .def(
            "add_polygon_from_segments_and_curves",
            &wrapped_type::add_polygon_from_segments_and_curves,
            py::arg("segments"),
            py::arg("curves"),
            py::arg("sample_length"))
        .def("get_polygon", &wrapped_type::get_polygon, py::arg("polygon_id"))
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
        .def(
            "decompose_to_trapezoid",
            [](wrapped_type & self, size_t polygon_id)
            {
                return self.decompose_to_trapezoid(polygon_id);
            },
            py::arg("polygon_id"))
        .def(
            "boolean_union",
            &wrapped_type::boolean_union,
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "boolean_intersection",
            &wrapped_type::boolean_intersection,
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "boolean_difference",
            &wrapped_type::boolean_difference,
            py::arg("p1"),
            py::arg("p2"));
}

void wrap_polygon(pybind11::module & mod)
{
    WrapPolygon<float>::commit(mod, "Polygon3dFp32", "Polygon3dFp32");
    WrapPolygon<double>::commit(mod, "Polygon3dFp64", "Polygon3dFp64");

    WrapPolygonPad<float>::commit(mod, "PolygonPadFp32", "PolygonPadFp32");
    WrapPolygonPad<double>::commit(mod, "PolygonPadFp64", "PolygonPadFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
