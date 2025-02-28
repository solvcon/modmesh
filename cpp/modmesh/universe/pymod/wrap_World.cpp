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
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPoint3d
    : public WrapBase<WrapPoint3d<T>, Point3d<T>>
{

public:

    using base_type = WrapBase<WrapPoint3d<T>, Point3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapPoint3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapPoint3d */

template <typename T>
WrapPoint3d<T>::WrapPoint3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<value_type, value_type, value_type>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(
            "__str__",
            [](wrapped_type const & self)
            {
                return (Formatter()
                        << "Vector3d(" << self.x() << ", " << self.y() << ", " << self.z() << ")")
                    .str();
            })
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
            [](wrapped_type & self, size_t it, value_type val)
            { self.at(it) = val; })
        .def("fill", &wrapped_type::fill, py::arg("value"))
        //
        ;

    // Wrap for operators.
    (*this)
        .def(py::self += py::self)
        .def(py::self += value_type())
        .def(py::self -= py::self)
        .def(py::self -= value_type())
        .def(py::self *= value_type())
        .def(py::self /= value_type())
        //
        ;

    // x, y, z accessors.
#define DECL_WRAP(NAME)                                              \
    .def_property(                                                   \
        #NAME,                                                       \
        [](wrapped_type const & self)                                \
        { return self.NAME(); },                                     \
        [](wrapped_type & self, typename wrapped_type::value_type v) \
        { self.NAME() = v; })
    // clang-format off
   (*this)
       DECL_WRAP(x)
       DECL_WRAP(y)
       DECL_WRAP(z)
       ;
#undef DECL_WRAP
    // clang-format on
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPointPad
    : public WrapBase<WrapPointPad<T>, PointPad<T>, std::shared_ptr<PointPad<T>>>
{

public:

    using base_type = WrapBase<WrapPointPad<T>, PointPad<T>, std::shared_ptr<PointPad<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename base_type::wrapped_type::value_type;
    using point_type = typename base_type::wrapped_type::point_type;

    friend typename base_type::root_base_type;

protected:

    WrapPointPad(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapPointPad */

template <typename T>
WrapPointPad<T>::WrapPointPad(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
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
                [](SimpleArray<T> & x, SimpleArray<T> & y, bool clone)
                { return wrapped_type::construct(x, y, clone); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("clone"))
        .def(
            py::init(
                [](SimpleArray<T> & x, SimpleArray<T> & y, SimpleArray<T> & z, bool clone)
                { return wrapped_type::construct(x, y, z, clone); }),
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::arg("clone"))
        //
        ;

    (*this)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def(
            "append",
            [](wrapped_type & self, value_type x, value_type y)
            {
                self.append(x, y);
            },
            py::arg("x"),
            py::arg("y"))
        .def(
            "append",
            [](wrapped_type & self, value_type x, value_type y, value_type z)
            {
                self.append(x, y, z);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def_timed("pack_array", &wrapped_type::pack_array)
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        .def("__len__", &wrapped_type::size)
        .def("__getitem__",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("get_at",
             [](wrapped_type const & self, size_t it)
             {
                 return self.get_at(it);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, point_type const & v)
             {
                 self.set_at(it, v);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x, value_type y)
             {
                 self.set_at(it, x, y);
             })
        .def("set_at",
             [](wrapped_type & self, size_t it, value_type x, value_type y, value_type z)
             {
                 self.set_at(it, x, y, z);
             })
        //
        ;

    // x, y, z element accessors.
#define DECL_WRAP(NAME)                         \
    .def(                                       \
        #NAME,                                  \
        [](wrapped_type const & self, size_t i) \
        { return self.NAME(i); })
    // clang-format off
    (*this)
        DECL_WRAP(x_at)
        DECL_WRAP(y_at)
        DECL_WRAP(z_at)
        ;
#undef DECL_WRAP
    // clang-format on

    // x, y, z array/batch accessors.
#define DECL_WRAP(NAME)         \
    .def_property_readonly(     \
        #NAME,                  \
        [](wrapped_type & self) \
        { return self.NAME(); })
    // clang-format off
    (*this)
        DECL_WRAP(x)
        DECL_WRAP(y)
        DECL_WRAP(z)
        ;
#undef DECL_WRAP
    // clang-format on
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSegment3d
    : public WrapBase<WrapSegment3d<T>, Segment3d<T>>
{

public:

    using base_type = WrapBase<WrapSegment3d<T>, Segment3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;
    using value_type = typename wrapped_type::value_type;

    friend typename base_type::root_base_type;

protected:

    WrapSegment3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapSegment3d */

template <typename T>
WrapSegment3d<T>::WrapSegment3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<point_type const &, point_type const &>(),
             py::arg("v0"),
             py::arg("v1"))
        .def(py::init<value_type, value_type, value_type, value_type, value_type, value_type>(),
             py::arg("x0"),
             py::arg("y0"),
             py::arg("z0"),
             py::arg("x1"),
             py::arg("y1"),
             py::arg("z1"))
        .def(
            "__str__",
            [](wrapped_type const & self)
            {
                return (Formatter()
                        << "Edge3d("
                        << self.x0() << ", " << self.y0() << ", " << self.z0() << ", "
                        << self.x1() << ", " << self.y1() << ", " << self.z1() << ")")
                    .str();
            })
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

    // v0 (tail) and v1 (head) accessors.
#define DECL_WRAP(NAME)                               \
    .def_property(                                    \
        #NAME,                                        \
        [](wrapped_type const & self)                 \
        { return self.NAME(); },                      \
        [](wrapped_type & self, point_type const & v) \
        { self.NAME() = v; })
    // clang-format off
    (*this)
        DECL_WRAP(v0)
        DECL_WRAP(v1)
        ;
#undef DECL_WRAP
    // clang-format on

    // x, y, z accessors.
#define DECL_WRAP(NAME)                                              \
    .def_property(                                                   \
        #NAME,                                                       \
        [](wrapped_type const & self)                                \
        { return self.NAME(); },                                     \
        [](wrapped_type & self, typename wrapped_type::value_type v) \
        { self.NAME() = v; })
    // clang-format off
   (*this)
       DECL_WRAP(x0)
       DECL_WRAP(y0)
       DECL_WRAP(z0)
       DECL_WRAP(x1)
       DECL_WRAP(y1)
       DECL_WRAP(z1)
       ;
#undef DECL_WRAP
    // clang-format on
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapBezier3d
    : public WrapBase<WrapBezier3d<T>, Bezier3d<T>>
{

public:

    using base_type = WrapBase<WrapBezier3d<T>, Bezier3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    using point_type = typename wrapped_type::point_type;

    friend typename base_type::root_base_type;

protected:

    WrapBezier3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapBezier3d */

template <typename T>
WrapBezier3d<T>::WrapBezier3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<std::vector<point_type> const &>(), py::arg("controls"))
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
            [](wrapped_type & self, size_t it, point_type val)
            { self.at(it) = val; })
        //
        ;

    // Control points
    (*this)
        .def_property(
            "control_points",
            [](wrapped_type const & self)
            {
                std::vector<point_type> ret(self.ncontrol());
                for (size_t i = 0; i < self.ncontrol(); ++i)
                {
                    ret[i] = self.control(i);
                }
                return ret;
            },
            [](wrapped_type & self, std::vector<point_type> const & points)
            {
                if (points.size() != self.ncontrol())
                {
                    throw std::out_of_range(
                        Formatter() << "Bezier3d.control_points: len(points) " << points.size() << " != ncontrol " << self.ncontrol());
                }
                for (size_t i = 0; i < self.ncontrol(); ++i)
                {
                    self.control(i) = points[i];
                }
            })
        //
        ;

    // Locus points
    (*this)
        .def("sample", &wrapped_type::sample, py::arg("nlocus"))
        .def_property_readonly("nlocus", &wrapped_type::nlocus)
        .def_property_readonly(
            "locus_points",
            [](wrapped_type const & self)
            {
                std::vector<typename wrapped_type::point_type> ret(self.nlocus());
                for (size_t i = 0; i < self.nlocus(); ++i)
                {
                    ret[i] = self.locus(i);
                }
                return ret;
            })
        //
        ;
}

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapWorld
    : public WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>
{

public:

    using base_type = WrapBase<WrapWorld<T>, World<T>, std::shared_ptr<World<T>>>;
    using wrapped_type = typename base_type::wrapped_type;

    using value_type = typename wrapped_type::value_type;
    using vector_type = typename wrapped_type::vector_type;
    using vertex_type = typename wrapped_type::vertex_type;
    using segment_type = typename wrapped_type::segment_type;
    using bezier_type = typename wrapped_type::bezier_type;

    friend typename base_type::root_base_type;

protected:

    WrapWorld(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapWorld */

template <typename T>
WrapWorld<T>::WrapWorld(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                []()
                { return wrapped_type::construct(); }))
        //
        ;

    // Bezier curves
    (*this)
        .def(
            "add_vertex",
            [](wrapped_type & self, vertex_type const & vertex) -> auto &
            {
                self.add_vertex(vertex);
                return self.vertex_at(self.nvertex() - 1);
            },
            py::arg("vertex"),
            py::return_value_policy::reference_internal)
        .def(
            "add_vertex",
            [](wrapped_type & self, value_type x, value_type y, value_type z) -> auto &
            {
                self.add_vertex(x, y, z);
                return self.vertex_at(self.nvertex() - 1);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            py::return_value_policy::reference_internal)
        .def_property_readonly("nvertex", &wrapped_type::nvertex)
        .def(
            "vertex",
            [](wrapped_type & self, size_t i) -> auto &
            {
                return self.vertex_at(i);
            },
            py::return_value_policy::reference_internal)
        .def(
            "add_segment",
            [](wrapped_type & self, segment_type const & edge) -> auto &
            {
                self.add_segment(edge);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("edge"),
            py::return_value_policy::reference_internal)
        .def(
            "add_segment",
            [](wrapped_type & self, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1) -> auto &
            {
                self.add_segment(x0, y0, z0, x1, y1, z1);
                return self.segment_at(self.nsegment() - 1);
            },
            py::arg("x0"),
            py::arg("y0"),
            py::arg("z0"),
            py::arg("x1"),
            py::arg("y1"),
            py::arg("z1"),
            py::return_value_policy::reference_internal)
        .def_property_readonly("nsegment", &wrapped_type::nsegment)
        .def(
            "edge",
            [](wrapped_type & self, size_t i) -> auto &
            {
                return self.segment_at(i);
            },
            py::return_value_policy::reference_internal)
        .def(
            "add_bezier",
            [](wrapped_type & self, std::vector<typename wrapped_type::vector_type> const & controls) -> auto &
            {
                self.add_bezier(controls);
                return self.bezier_at(self.nbezier() - 1);
            },
            py::arg("controls"),
            py::return_value_policy::reference_internal)
        .def_property_readonly("nbezier", &wrapped_type::nbezier)
        .def(
            "bezier",
            [](wrapped_type & self, size_t i) -> auto &
            {
                return self.bezier_at(i);
            },
            py::return_value_policy::reference_internal)
        //
        ;
}

void wrap_World(pybind11::module & mod)
{
    WrapPoint3d<float>::commit(mod, "Point3dFp32", "Point3dFp32");
    WrapPoint3d<double>::commit(mod, "Point3dFp64", "Point3dFp64");
    WrapPointPad<float>::commit(mod, "PointPadFp32", "PointPadFp32");
    WrapPointPad<double>::commit(mod, "PointPadFp64", "PointPadFp64");
    WrapSegment3d<float>::commit(mod, "Segment3dFp32", "Segment3dFp32");
    WrapSegment3d<double>::commit(mod, "Segment3dFp64", "Segment3dFp64");
    WrapBezier3d<float>::commit(mod, "Bezier3dFp32", "Bezier3dFp32");
    WrapBezier3d<double>::commit(mod, "Bezier3dFp64", "Bezier3dFp64");
    WrapWorld<float>::commit(mod, "WorldFp32", "WorldFp32");
    WrapWorld<double>::commit(mod, "WorldFp64", "WorldFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
