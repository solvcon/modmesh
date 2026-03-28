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
#include <pybind11/operators.h>

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapWorld
    : public WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>
{

public:

    using base_type = WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename wrapped_type::value_type;
    using point_type = typename wrapped_type::point_type;
    using segment_type = typename wrapped_type::segment_type;
    using segment_pad_type = typename wrapped_type::segment_pad_type;
    using bezier_type = typename wrapped_type::bezier_type;
    using curve_pad_type = typename wrapped_type::curve_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapWorld(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapWorld & wrap_management();
    WrapWorld & wrap_point();
    WrapWorld & wrap_segment();
    WrapWorld & wrap_bezier();
};
/* end class WrapWorld */

template <typename T>
WrapWorld<T>::WrapWorld(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_point()
        .wrap_segment()
        .wrap_bezier()
        //
        ;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                []()
                { return wrapped_type::construct(); }))
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_point()
{
    namespace py = pybind11;

    // Point.
    (*this)
        .def(
            "add_point",
            [](wrapped_type & self, point_type const & point)
            {
                self.add_point(point);
                return self.point_at(self.npoint() - 1);
            },
            py::arg("point"))
        .def(
            "add_point",
            [](wrapped_type & self, value_type x, value_type y, value_type z)
            {
                self.add_point(x, y, z);
                return self.point_at(self.npoint() - 1);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def_property_readonly("npoint", &wrapped_type::npoint)
        .def("point", &wrapped_type::point_at)
        .def_property_readonly("points", &wrapped_type::points)
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_segment()
{
    namespace py = pybind11;

    // Segment.
    (*this)
        .def(
            "add_segment",
            [](wrapped_type & self, segment_type const & edge)
            {
                self.add_segment(edge);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("s"))
        .def(
            "add_segment",
            [](wrapped_type & self, point_type const & p0, point_type const & p1)
            {
                self.add_segment(p0, p1);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("p0"),
            py::arg("p1"))
        .def(
            "add_segments",
            [](wrapped_type & self, segment_pad_type const & segment_pad)
            {
                for (size_t i = 0; i < segment_pad.size(); ++i)
                {
                    self.add_segment(segment_pad.get(i));
                }
            },
            py::arg("pad"))
        .def_property_readonly("nsegment", &wrapped_type::nsegment)
        .def("segment", &wrapped_type::segment_at)
        .def_property_readonly("segments", &wrapped_type::segments)
        //
        ;

    return *this;
}

template <typename T>
WrapWorld<T> & WrapWorld<T>::wrap_bezier()
{
    namespace py = pybind11;

    // Bezier curves
    (*this)
        .def(
            "add_bezier",
            [](wrapped_type & self, bezier_type const & bezier)
            {
                self.add_bezier(bezier);
                return self.bezier_at(self.nbezier() - 1);
            },
            py::arg("b"))
        .def(
            "add_bezier",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
            {
                self.add_bezier(p0, p1, p2, p3);
                return self.bezier_at(self.nbezier() - 1);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"),
            py::arg("p3"))
        .def(
            "add_beziers",
            [](wrapped_type & self, curve_pad_type const & curve_pad)
            {
                for (size_t i = 0; i < curve_pad.size(); ++i)
                {
                    self.add_bezier(curve_pad.get(i));
                }
            },
            py::arg("pad"))
        .def_property_readonly("nbezier", &wrapped_type::nbezier)
        .def(
            "bezier",
            [](wrapped_type & self, size_t i)
            {
                return self.bezier_at(i);
            })
        .def_property_readonly("curves", &wrapped_type::curves)
        //
        ;

    return *this;
}

void wrap_World(pybind11::module & mod)
{
    WrapWorld<float>::commit(mod, "WorldFp32", "WorldFp32");
    WrapWorld<double>::commit(mod, "WorldFp64", "WorldFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
