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
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapVector3d
    : public WrapBase<WrapVector3d<T>, Vector3d<T>>
{

public:

    using value_type = T;
    using base_type = WrapBase<WrapVector3d<T>, Vector3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend typename base_type::root_base_type;

protected:

    WrapVector3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapVector3dF32 */

template <typename T>
WrapVector3d<T>::WrapVector3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<value_type, value_type, value_type>(), py::arg("x"), py::arg("y"), py::arg("z"))
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
            [](wrapped_type & self, size_t it, typename wrapped_type::value_type val)
            { self.at(it) = val; })
        .def("fill", &wrapped_type::fill, py::arg("value"))
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
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapBezier3d
    : public WrapBase<WrapBezier3d<T>, Bezier3d<T>>
{

public:

    using vector_type = Vector3d<T>;
    using base_type = WrapBase<WrapBezier3d<T>, Bezier3d<T>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend typename base_type::root_base_type;

protected:

    WrapBezier3d(pybind11::module & mod, char const * pyname, char const * pydoc);
};
/* end class WrapBezier3dF32 */

template <typename T>
WrapBezier3d<T>::WrapBezier3d(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(py::init<std::vector<typename wrapped_type::vector_type> const &>(), py::arg("vectors"))
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
            [](wrapped_type & self, size_t it, typename wrapped_type::vector_type val)
            { self.at(it) = val; })
        //
        ;

    // Control points
    (*this)
        .def_property(
            "control_points",
            [](wrapped_type const & self)
            {
                std::vector<typename wrapped_type::vector_type> ret(self.ncontrol());
                for (size_t i = 0; i < self.ncontrol(); ++i)
                {
                    ret[i] = self.control(i);
                }
                return ret;
            },
            [](wrapped_type & self, std::vector<typename wrapped_type::vector_type> const & points)
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
        .def_property_readonly(
            "locus_points",
            [](wrapped_type const & self)
            {
                std::vector<typename wrapped_type::vector_type> ret(self.nlocus());
                for (size_t i = 0; i < self.nlocus(); ++i)
                {
                    ret[i] = self.locus(i);
                }
                return ret;
            })
        //
        ;
}

void wrap_World(pybind11::module & mod)
{
    WrapVector3d<float>::commit(mod, "Vector3dFp32", "Vector3dFp32");
    WrapVector3d<double>::commit(mod, "Vector3dFp64", "Vector3dFp64");
    WrapBezier3d<float>::commit(mod, "Bezier3dFp32", "Bezier3dFp32");
    WrapBezier3d<double>::commit(mod, "Bezier3dFp64", "Bezier3dFp64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
