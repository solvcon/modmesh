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

#include <modmesh/buffer/pymod/buffer_pymod.hpp> // Must be the first include.

#include <modmesh/buffer/pymod/array_common.hpp>

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSimpleArray
    : public WrapBase<WrapSimpleArray<T>, SimpleArray<T>>
{

    using root_base_type = WrapBase<WrapSimpleArray<T>, SimpleArray<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using wrapper_type = typename root_base_type::wrapper_type;
    using value_type = typename wrapped_type::value_type;
    using property_helper = ArrayPropertyHelper<T>;

    friend root_base_type;

    WrapSimpleArray(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
    {
        namespace py = pybind11;

        (*this)
            .def_timed(
                py::init(
                    [](py::object const & shape)
                    { return wrapped_type(make_shape(shape)); }),
                py::arg("shape"))
            .def_timed(
                py::init(
                    [](py::object const & shape, value_type const & value)
                    { return wrapped_type(make_shape(shape), value); }),
                py::arg("shape"),
                py::arg("value"))
            .def(
                py::init(
                    [](py::array & arr_in)
                    {
                        if (!dtype_is_type<T>(arr_in))
                        {
                            throw std::runtime_error("dtype mismatch");
                        }

                        modmesh::detail::shape_type shape;
                        modmesh::detail::shape_type stride;
                        constexpr size_t itemsize = wrapped_type::itemsize();
                        for (ssize_t i = 0; i < arr_in.ndim(); ++i)
                        {
                            shape.push_back(arr_in.shape(i));
                            stride.push_back(arr_in.strides(i) / itemsize);
                        }

                        const bool is_c_contiguous = (arr_in.flags() & py::array::c_style) == py::array::c_style;
                        const bool is_f_contiguous = (arr_in.flags() & py::array::f_style) == py::array::f_style;

                        std::shared_ptr<ConcreteBuffer> const buffer = ConcreteBuffer::construct(
                            arr_in.nbytes(),
                            arr_in.mutable_data(),
                            std::make_unique<ConcreteBufferNdarrayRemover>(arr_in));
                        return wrapped_type(shape, stride, buffer, is_c_contiguous, is_f_contiguous);
                    }),
                py::arg("array"))
            .def_buffer(&property_helper::get_buffer_info)
            .def("clone",
                 [](wrapped_type const & self)
                 { return wrapped_type(self); }) // cloning the object using the copy constructor. never add the clone method to the C++ class.
            .def_property_readonly(
                "ndarray",
                [](wrapped_type & self)
                { return to_ndarray(self); })
            .def_property_readonly(
                "is_from_python",
                [](wrapped_type const & self)
                {
                    return self.buffer().has_remover() && ConcreteBufferNdarrayRemover::is_same_type(self.buffer().get_remover());
                })
            .def_property_readonly("nbytes", &wrapped_type::nbytes)
            .def_property_readonly("size", &wrapped_type::size)
            .def_property_readonly("itemsize", &wrapped_type::itemsize)
            .def_property_readonly(
                "shape",
                [](wrapped_type const & self)
                {
                    py::tuple ret(self.shape().size());
                    for (size_t i = 0; i < self.shape().size(); ++i)
                    {
                        ret[i] = self.shape()[i];
                    }
                    return ret;
                })
            .def_property_readonly(
                "stride",
                [](wrapped_type const & self)
                {
                    py::tuple ret(self.stride().size());
                    for (size_t i = 0; i < self.stride().size(); ++i)
                    {
                        ret[i] = self.stride()[i];
                    }
                    return ret;
                })
            .def("__len__", &wrapped_type::size)
            .def(
                "__getitem__",
                [](wrapped_type const & self, ssize_t key)
                { return self.at(key); })
            .def(
                "__getitem__",
                [](wrapped_type const & self, std::vector<ssize_t> const & key)
                { return self.at(key); })
            .def("__setitem__", &property_helper::setitem_parser)
            .def(
                "reshape",
                [](wrapped_type const & self, py::object const & shape)
                { return self.reshape(make_shape(shape)); })
            .def_property_readonly("has_ghost", &wrapped_type::has_ghost)
            .def_property("nghost", &wrapped_type::nghost, &wrapped_type::set_nghost)
            .def_property_readonly("nbody", &wrapped_type::nbody)
            .def_property_readonly("plex", [](wrapped_type const & arr)
                                   { return pybind11::cast(SimpleArrayPlex(arr)); })
            .wrap_modifiers()
            .wrap_calculators()
            .wrap_sort()
            // ATTENTION: always keep the same interface between WrapSimpleArrayPlex and WrapSimpleArray
            ;
    }

    wrapper_type & wrap_modifiers()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                // Use lambda to avoid bad implicit conversion of SimpleArray<T>in pybind11
                // (see https://github.com/solvcon/modmesh/issues/283)
                "fill",
                [](wrapped_type & arr, value_type const value)
                { arr.fill(value); },
                py::arg("value"))
            //
            ;

        return *this;
    }

    wrapper_type & wrap_calculators()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def("min", &wrapped_type::min)
            .def("max", &wrapped_type::max)
            .def("sum", &wrapped_type::sum)
            .def("abs", &wrapped_type::abs)
            //
            ;

        return *this;
    }

    wrapper_type & wrap_sort()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def("sort", &wrapped_type::sort)
            .def(
                "argsort",
                [](wrapped_type & self)
                { return pybind11::cast(self.argsort()); })
            .def("take_along_axis",
                 [](wrapped_type & self, py::object const & indices)
                 { return pybind11::cast(self.take_along_axis(indices.cast<SimpleArrayUint64>())); })
            //
            ;

        return *this;
    }
}; /* end class WrapSimpleArray */

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSimpleCollector
    : public WrapBase<WrapSimpleCollector<T>, SimpleCollector<T>>
{

    using root_base_type = WrapBase<WrapSimpleCollector<T>, SimpleCollector<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using wrapper_type = typename root_base_type::wrapper_type;
    using value_type = typename wrapped_type::value_type;

    friend root_base_type;

    WrapSimpleCollector(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapSimpleCollector */

template <typename T>
WrapSimpleCollector<T>::WrapSimpleCollector(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](size_t length)
                { return wrapped_type(length); }),
            py::arg("length"))
        .def_timed(py::init<>())
        .def_timed("reserve", &wrapped_type::reserve, py::arg("cap"))
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        .def_property_readonly("capacity", &wrapped_type::capacity)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t key)
            { return self.at(key); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t key, value_type val)
            { self.at(key) = val; })
        .def(
            "push_back",
            [](wrapped_type & self, value_type value)
            { self.push_back(value); })
        .def_timed("as_array", &wrapped_type::as_array)
        //
        ;
}

void wrap_SimpleArray(pybind11::module & mod)
{
    WrapSimpleArray<bool>::commit(mod, "SimpleArrayBool", "SimpleArrayBool");
    WrapSimpleArray<int8_t>::commit(mod, "SimpleArrayInt8", "SimpleArrayInt8");
    WrapSimpleArray<int16_t>::commit(mod, "SimpleArrayInt16", "SimpleArrayInt16");
    WrapSimpleArray<int32_t>::commit(mod, "SimpleArrayInt32", "SimpleArrayInt32");
    WrapSimpleArray<int64_t>::commit(mod, "SimpleArrayInt64", "SimpleArrayInt64");
    WrapSimpleArray<uint8_t>::commit(mod, "SimpleArrayUint8", "SimpleArrayUint8");
    WrapSimpleArray<uint16_t>::commit(mod, "SimpleArrayUint16", "SimpleArrayUint16");
    WrapSimpleArray<uint32_t>::commit(mod, "SimpleArrayUint32", "SimpleArrayUint32");
    WrapSimpleArray<uint64_t>::commit(mod, "SimpleArrayUint64", "SimpleArrayUint64");
    WrapSimpleArray<float>::commit(mod, "SimpleArrayFloat32", "SimpleArrayFloat32");
    WrapSimpleArray<double>::commit(mod, "SimpleArrayFloat64", "SimpleArrayFloat64");

    WrapSimpleCollector<bool>::commit(mod, "SimpleCollectorBool", "SimpleCollectorBool");
    WrapSimpleCollector<int8_t>::commit(mod, "SimpleCollectorInt8", "SimpleCollectorInt8");
    WrapSimpleCollector<int16_t>::commit(mod, "SimpleCollectorInt16", "SimpleCollectorInt16");
    WrapSimpleCollector<int32_t>::commit(mod, "SimpleCollectorInt32", "SimpleCollectorInt32");
    WrapSimpleCollector<int64_t>::commit(mod, "SimpleCollectorInt64", "SimpleCollectorInt64");
    WrapSimpleCollector<uint8_t>::commit(mod, "SimpleCollectorUint8", "SimpleCollectorUint8");
    WrapSimpleCollector<uint16_t>::commit(mod, "SimpleCollectorUint16", "SimpleCollectorUint16");
    WrapSimpleCollector<uint32_t>::commit(mod, "SimpleCollectorUint32", "SimpleCollectorUint32");
    WrapSimpleCollector<uint64_t>::commit(mod, "SimpleCollectorUint64", "SimpleCollectorUint64");
    WrapSimpleCollector<float>::commit(mod, "SimpleCollectorFloat32", "SimpleCollectorFloat32");
    WrapSimpleCollector<double>::commit(mod, "SimpleCollectorFloat64", "SimpleCollectorFloat64");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
