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
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapTriangle3d
    : public WrapBase<WrapTriangle3d<T>, Triangle3d<T>>
{

public:

    using base_type = WrapBase<WrapTriangle3d<T>, Triangle3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;
    using value_type = typename wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapTriangle3d(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapTriangle3d & wrap_management();
    WrapTriangle3d & wrap_operator();
    WrapTriangle3d & wrap_accessor();
    WrapTriangle3d & wrap_geometry();
};
/* end class WrapTriangle3d */

template <typename T>
WrapTriangle3d<T>::WrapTriangle3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_operator()
        .wrap_accessor()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(py::init<point_type const &, point_type const &, point_type const &>(),
             py::arg("p0"),
             py::arg("p1"),
             py::arg("p2"))
        .def(
            "__repr__",
            [](wrapped_type const & self)
            {
                static char const * ttypename = std::is_same_v<T, double> ? "Triangle3dFp64" : "Triangle3dFp32";
                static char const * ptypename = std::is_same_v<T, double> ? "Point3dFp64" : "Point3dFp32";
                return std::format("{}({}({}), {}({}), {}({}))",
                                   ttypename,
                                   ptypename,
                                   self.p0().value_string(),
                                   ptypename,
                                   self.p1().value_string(),
                                   ptypename,
                                   self.p2().value_string());
            })
        .def_alias("__repr__", "__str__")
        //
        ;

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_operator()
{
    namespace py = pybind11;

    (*this)
        .def(py::self == py::self) // NOLINT(misc-redundant-expression)
        .def(py::self != py::self) // NOLINT(misc-redundant-expression)
        //
        ;

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_accessor()
{
    namespace py = pybind11;

    (*this)
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
            [](wrapped_type & self, size_t it, point_type const & vec)
            { self.at(it) = vec; })
        //
        ;

#define DECL_WRAP(NAME)                               \
    .def_property(                                    \
        #NAME,                                        \
        [](wrapped_type const & self)                 \
        { return self.NAME(); },                      \
        [](wrapped_type & self, point_type const & v) \
        { self.set_##NAME(v); })
    // clang-format off
    (*this)
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        DECL_WRAP(p2);
    // clang-format on
#undef DECL_WRAP

#define DECL_WRAP(NAME)                                              \
    .def_property(                                                   \
        #NAME,                                                       \
        [](wrapped_type const & self)                                \
        { return self.NAME(); },                                     \
        [](wrapped_type & self, typename wrapped_type::value_type v) \
        { self.set_##NAME(v); })
    // clang-format off
    (*this)
        DECL_WRAP(x0)
        DECL_WRAP(y0)
        DECL_WRAP(z0)
        DECL_WRAP(x1)
        DECL_WRAP(y1)
        DECL_WRAP(z1)
        DECL_WRAP(x2)
        DECL_WRAP(y2)
        DECL_WRAP(z2);
    // clang-format on
#undef DECL_WRAP

    return *this;
}

template <typename T>
WrapTriangle3d<T> & WrapTriangle3d<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def(
            "mirror",
            [](wrapped_type & self, std::string const & axis)
            {
                if (axis == "x" || axis == "X")
                {
                    self.mirror(Axis::X);
                }
                else if (axis == "y" || axis == "Y")
                {
                    self.mirror(Axis::Y);
                }
                else if (axis == "z" || axis == "Z")
                {
                    self.mirror(Axis::Z);
                }
                else
                {
                    throw std::invalid_argument("Triangle3d::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapTrianglePad
    : public WrapBase<WrapTrianglePad<T>, TrianglePad<T>, std::shared_ptr<TrianglePad<T>>>
{

public:

    using base_type = WrapBase<WrapTrianglePad<T>, TrianglePad<T>, std::shared_ptr<TrianglePad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;
    using triangle_type = typename base_type::wrapped_type::triangle_type;
    using point_pad_type = typename base_type::wrapped_type::point_pad_type;

    friend typename base_type::root_base_type;

protected:

    WrapTrianglePad(pybind11::module & mod, char const * pyname, char const * pydoc);

    WrapTrianglePad & wrap_management();
    WrapTrianglePad & wrap_accessor_triangle();
    WrapTrianglePad & wrap_accessor_point();
    WrapTrianglePad & wrap_geometry();
}; /* end class WrapTrianglePad */

template <typename T>
WrapTrianglePad<T>::WrapTrianglePad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .wrap_management()
        .wrap_accessor_triangle()
        .wrap_accessor_point()
        .wrap_geometry()
        //
        ;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_management()
{
    namespace py = pybind11;

    // Constructors
    (*this)
        .def(
            py::init(
                [](uint8_t ndim)
                { return wrapped_type::construct(ndim); }),
            py::arg("ndim"))
        .def(
            py::init(
                [](uint8_t ndim, size_t nelem)
                { return wrapped_type::construct(ndim, nelem); }),
            py::arg("ndim"),
            py::arg("nelem"))
        .def(
            py::init(
                [](
                    SimpleArray<T> & x0,
                    SimpleArray<T> & y0,
                    SimpleArray<T> & x1,
                    SimpleArray<T> & y1,
                    SimpleArray<T> & x2,
                    SimpleArray<T> & y2,
                    bool clone)
                { return wrapped_type::construct(x0, y0, x1, y1, x2, y2, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("clone"))
        .def(
            py::init(
                [](
                    SimpleArray<T> & x0,
                    SimpleArray<T> & y0,
                    SimpleArray<T> & z0,
                    SimpleArray<T> & x1,
                    SimpleArray<T> & y1,
                    SimpleArray<T> & z1,
                    SimpleArray<T> & x2,
                    SimpleArray<T> & y2,
                    SimpleArray<T> & z2,
                    bool clone)
                { return wrapped_type::construct(x0, y0, z0, x1, y1, z1, x2, y2, z2, clone); }),
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("z2"),
            py::arg("clone"))
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_timed("clone", &wrapped_type::clone)
        .def_timed("pack_array", &wrapped_type::pack_array)
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        //
        ;

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_accessor_triangle()
{
    namespace py = pybind11;

    // Python dunder accessor.
    (*this)
        .def("__len__", &wrapped_type::size)
        .def("__getitem__",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        //
        ;

    // Append Triangle3d.
    (*this)
        .def(
            "append",
            [](wrapped_type & self, triangle_type const & t)
            {
                self.append(t);
            },
            py::arg("triangle"))
        .def(
            "append",
            [](wrapped_type & self, point_type const & p0, point_type const & p1, point_type const & p2)
            {
                self.append(p0, p1, p2);
            },
            py::arg("p0"),
            py::arg("p1"),
            py::arg("p2"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
            {
                self.append(x0, y0, x1, y1, x2, y2);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("x2"),
            py::arg("y2"))
        .def(
            "append",
            [](wrapped_type & self, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
            {
                self.append(x0, y0, z0, x1, y1, z1, x2, y2, z2);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::arg("x2"),
            py::arg("y2"),
            py::arg("z2"))
        //
        ;

    // Extend Triangle3d
    (*this)
        .def(
            "extend_with",
            [](wrapped_type & self, wrapped_type const & other)
            {
                self.extend_with(other);
            },
            py::arg("triangles"))
        //
        ;

    // Additional getter and setter for Triangle3d.
    (*this)
        .def("get_at",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, triangle_type const & t)
             {
                 self.set_at(it, t);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, point_type const & p0, point_type const & p1, point_type const & p2)
             {
                 self.set_at(it, p0, p1, p2);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
             {
                 self.set_at(it, x0, y0, x1, y1, x2, y2);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
             {
                 self.set_at(it, x0, y0, z0, x1, y1, z1, x2, y2, z2);
             })
        //
        ;

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_accessor_point()
{
    namespace py = pybind11;

    // x, y, z, point element accessors.
#define DECL_WRAP(NAME)                         \
    .def(                                       \
        #NAME,                                  \
        [](wrapped_type const & self, size_t i) \
        { return self.NAME(i); })
    // clang-format off
    (*this)
        DECL_WRAP(x0_at)
        DECL_WRAP(y0_at)
        DECL_WRAP(z0_at)
        DECL_WRAP(x1_at)
        DECL_WRAP(y1_at)
        DECL_WRAP(z1_at)
        DECL_WRAP(x2_at)
        DECL_WRAP(y2_at)
        DECL_WRAP(z2_at)
        DECL_WRAP(p0_at)
        DECL_WRAP(p1_at)
        DECL_WRAP(p2_at);
    // clang-format on
#undef DECL_WRAP

    // x, y, z, point array/batch accessors.
#define DECL_WRAP(NAME)         \
    .def_property_readonly(     \
        #NAME,                  \
        [](wrapped_type & self) \
        { return self.NAME(); })
    // clang-format off
    (*this)
        DECL_WRAP(x0)
        DECL_WRAP(y0)
        DECL_WRAP(z0)
        DECL_WRAP(x1)
        DECL_WRAP(y1)
        DECL_WRAP(z1)
        DECL_WRAP(x2)
        DECL_WRAP(y2)
        DECL_WRAP(z2)
        DECL_WRAP(p0)
        DECL_WRAP(p1)
        DECL_WRAP(p2);
    // clang-format on
#undef DECL_WRAP

    return *this;
}

template <typename T>
WrapTrianglePad<T> & WrapTrianglePad<T>::wrap_geometry()
{
    namespace py = pybind11;

    (*this)
        .def(
            "mirror",
            [](wrapped_type & self, std::string const & axis)
            {
                if (axis == "x" || axis == "X")
                {
                    self.mirror(Axis::X);
                }
                else if (axis == "y" || axis == "Y")
                {
                    self.mirror(Axis::Y);
                }
                else if (axis == "z" || axis == "Z")
                {
                    self.mirror(Axis::Z);
                }
                else
                {
                    throw std::invalid_argument("TrianglePad::mirror: axis must be 'x', 'y', or 'z'");
                }
            },
            py::arg("axis"))
        //
        ;

    return *this;
}

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
    WrapBoundBox3d<float>::commit(mod, "BoundBox3dFp32", "BoundBox3dFp32");
    WrapBoundBox3d<double>::commit(mod, "BoundBox3dFp64", "BoundBox3dFp64");
    WrapTriangle3d<float>::commit(mod, "Triangle3dFp32", "Triangle3dFp32");
    WrapTriangle3d<double>::commit(mod, "Triangle3dFp64", "Triangle3dFp64");
    WrapTrianglePad<float>::commit(mod, "TrianglePadFp32", "TrianglePadFp32");
    WrapTrianglePad<double>::commit(mod, "TrianglePadFp64", "TrianglePadFp64");
    WrapWorld<float>::commit(mod, "WorldFp32", "WorldFp32");
    WrapWorld<double>::commit(mod, "WorldFp64", "WorldFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
