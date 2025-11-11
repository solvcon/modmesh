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
                    [](py::object const & shape, size_t alignment)
                    { return wrapped_type(make_shape(shape), alignment, with_alignment_t{}); }),
                py::arg("shape"),
                py::arg("alignment"))
            .def_timed(
                py::init(
                    [](py::object const & shape, value_type const & value)
                    { return wrapped_type(make_shape(shape), value); }),
                py::arg("shape"),
                py::arg("value"))
            .def_timed(
                py::init(
                    [](py::object const & shape, value_type const & value, size_t alignment)
                    { return wrapped_type(make_shape(shape), value, alignment); }),
                py::arg("shape"),
                py::arg("value"),
                py::arg("alignment"))
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
                        constexpr size_t span = 0;
                        for (ssize_t i = 0; i < arr_in.ndim(); ++i)
                        {
                            shape.push_back(arr_in.shape(i));
                            stride.push_back(arr_in.strides(i) / itemsize);
                        }

                        const bool is_c_contiguous = (arr_in.flags() & py::array::c_style) == py::array::c_style;
                        const bool is_f_contiguous = (arr_in.flags() & py::array::f_style) == py::array::f_style;

                        py::array owner = arr_in;
                        /*
                         * In the following document, it introduces the base object in ndarray.
                         * https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.base.html
                         * The `array.base` is base object if memory is from some other object.
                         * If object owns its memory, base is None.
                         */
                        while (true)
                        {
                            const py::object b = owner.attr("base");
                            if (b.is_none() || !py::isinstance<py::array>(b))
                            {
                                break;
                            }
                            auto next = b.cast<py::array>();
                            /*
                             * Prevent the infinite loop.
                             * For example, the following code will create a loop:
                             * nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
                             * nparr = nparr[::2, ::2, ::2]
                             */
                            if (next.ptr() == owner.ptr())
                            {
                                break;
                            }
                            owner = next;
                        }

                        char const * base_ptr = static_cast<char *>(owner.mutable_data());
                        char * view_ptr = static_cast<char *>(arr_in.mutable_data());
                        const ptrdiff_t offset_bytes = view_ptr - base_ptr;
                        if (offset_bytes < 0)
                        {
                            throw std::runtime_error("Unexpected negative offset!");
                        }
                        const size_t true_owner_nbytes = owner.nbytes();
                        const size_t view_nbytes = true_owner_nbytes - static_cast<size_t>(offset_bytes);
                        auto remover = std::make_unique<ConcreteBufferNdarrayRemover>(owner);
                        const std::shared_ptr<ConcreteBuffer> buffer =
                            ConcreteBuffer::construct(
                                view_nbytes,
                                view_ptr,
                                std::move(remover));
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
            .def_property_readonly("alignment", &wrapped_type::alignment)
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
            .def(
                "transpose",
                [](wrapped_type & self, py::object const & axis, bool const & inplace)
                {
                    wrapped_type * ret = inplace ? &self : new wrapped_type(self);
                    if (axis.is_none())
                    {
                        ret->transpose();
                    }
                    else
                    {
                        ret->transpose(make_shape(axis));
                    }
                    return *ret;
                },
                py::arg("axis") = py::none(),
                py::arg("inplace") = true)
            .def_property_readonly(
                "T",
                [](wrapped_type & self)
                {
                    auto ret = wrapped_type(self);
                    ret.transpose();
                    return ret;
                })
            .def_property_readonly("has_ghost", &wrapped_type::has_ghost)
            .def_property("nghost", &wrapped_type::nghost, &wrapped_type::set_nghost)
            .def_property_readonly("nbody", &wrapped_type::nbody)
            .def_property_readonly("plex", [](wrapped_type const & arr)
                                   { return pybind11::cast(SimpleArrayPlex(arr)); })
            .wrap_modifiers()
            .wrap_calculators()
            .wrap_sort()
            .wrap_search()
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
            .def(
                "median",
                [](wrapped_type const & self)
                {
                    return self.median();
                })
            .def(
                "median",
                [](wrapped_type const & self, py::object const & axis)
                { return self.median(make_shape(axis)); },
                py::arg("axis"))
            .def(
                "average",
                [](wrapped_type const & self, py::object const & weight)
                {
                    if (weight.is_none())
                    {
                        return self.mean();
                    }
                    auto w = weight.cast<wrapped_type>();
                    return self.average(w);
                },
                py::arg("weight") = py::none())
            .def(
                "average",
                &WrapSimpleArray<T>::average_with_axis,
                py::arg("axis"),
                py::arg("weight") = py::none())
            .def("mean",
                 [](wrapped_type const & self)
                 { return self.mean(); })
            .def(
                "mean",
                [](wrapped_type const & self, py::object const & axis)
                { return self.mean(make_shape(axis)); },
                py::arg("axis"))
            .def(
                "var",
                [](wrapped_type const & self, size_t ddof)
                { return self.var(ddof); },
                py::arg("ddof") = 0)
            .def(
                "var",
                [](wrapped_type const & self, py::object const & axis, size_t ddof)
                { return self.var(make_shape(axis), ddof); },
                py::arg("axis"),
                py::arg("ddof") = 0)
            .def(
                "std",
                [](wrapped_type const & self, size_t ddof)
                { return self.std(ddof); },
                py::arg("ddof") = 0)
            .def(
                "std",
                [](wrapped_type const & self, py::object const & axis, size_t ddof)
                { return self.std(make_shape(axis), ddof); },
                py::arg("axis"),
                py::arg("ddof") = 0)
            .def("min", &wrapped_type::min)
            .def("max", &wrapped_type::max)
            .def("sum", &wrapped_type::sum)
            .def("abs", &wrapped_type::abs)
            .def(
                "add",
                [](wrapped_type const & self, wrapped_type const & other)
                { return self.add(other); })
            .def(
                "add",
                [](wrapped_type const & self, value_type scalar)
                { return self.add(scalar); })
            .def(
                "sub",
                [](wrapped_type const & self, wrapped_type const & other)
                { return self.sub(other); })
            .def(
                "sub",
                [](wrapped_type const & self, value_type scalar)
                { return self.sub(scalar); })
            .def(
                "mul",
                [](wrapped_type const & self, wrapped_type const & other)
                { return self.mul(other); })
            .def(
                "mul",
                [](wrapped_type const & self, value_type scalar)
                { return self.mul(scalar); })
            .def("div", &wrapped_type::div)
            .def("matmul", &wrapped_type::matmul)
            .def("__matmul__", &wrapped_type::matmul)
            .def_static("eye", &wrapped_type::eye, py::arg("n"), "Create an identity matrix of size n x n")
            // TODO: In-place operation should return reference to self to support function chaining
            /*
             * Regular in-place methods (iadd, imul, etc.) are procedural calls and do
             * NOT need to return self. However, special __i*__ methods (__iadd__,
             * __imatmul__, etc.) MUST return self because they implement Python's
             * augmented assignment operators (+=, @=, etc.), which work via re-assignment
             * (e.g., a = a.__iadd__(b)).
             * See: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
             */
            .def(
                "iadd",
                [](wrapped_type & self, wrapped_type const & other)
                { self.iadd(other); })
            .def(
                "iadd",
                [](wrapped_type & self, value_type scalar)
                { self.iadd(scalar); })
            .def(
                "isub",
                [](wrapped_type & self, wrapped_type const & other)
                { self.isub(other); })
            .def(
                "isub",
                [](wrapped_type & self, value_type scalar)
                { self.isub(scalar); })
            .def(
                "imul",
                [](wrapped_type & self, wrapped_type const & other)
                { self.imul(other); })
            .def(
                "imul",
                [](wrapped_type & self, value_type scalar)
                { self.imul(scalar); })
            .def("idiv", [](wrapped_type & self, wrapped_type const & other)
                 { self.idiv(other); })
            .def("imatmul", [](wrapped_type & self, wrapped_type const & other)
                 { self.imatmul(other); })
            .def("__imatmul__", [](wrapped_type & self, wrapped_type const & other)
                 {
                     self.imatmul(other);
                     return self; })
            .def("add_simd", &wrapped_type::add_simd)
            .def("sub_simd", &wrapped_type::sub_simd)
            .def("mul_simd", &wrapped_type::mul_simd)
            .def("div_simd", &wrapped_type::div_simd)
            .def("iadd_simd", [](wrapped_type & self, wrapped_type const & other)
                 { self.iadd_simd(other); })
            .def("isub_simd", [](wrapped_type & self, wrapped_type const & other)
                 { self.isub_simd(other); })
            .def("imul_simd", [](wrapped_type & self, wrapped_type const & other)
                 { self.imul_simd(other); })
            .def("idiv_simd", [](wrapped_type & self, wrapped_type const & other)
                 { self.idiv_simd(other); })
            //
            ;

        return *this;
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static wrapped_type average_with_axis(wrapped_type const & self,
                                          pybind11::object const & axis,
                                          pybind11::object const & weight)
    {
        auto ashape = make_shape(axis);
        if (weight.is_none())
        {
            return self.mean(ashape);
        }
        auto w = weight.cast<wrapped_type>();
        return self.average(ashape, w);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    wrapper_type & wrap_sort()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def("sort", &wrapped_type::sort)
            .def(
                "argsort",
                [](wrapped_type & self)
                { return py::cast(self.argsort()); })
            .def("take_along_axis", &take_along_axis)
            .def("take_along_axis_simd", &take_along_axis_simd)
            //
            ;

        return *this;
    }

    static pybind11::object take_along_axis(wrapped_type & self, pybind11::object const & indices)
    {
        std::string py_typename(pybind11::detail::obj_class_name(indices.ptr()));
        const std::size_t found = py_typename.find("_modmesh.SimpleArray");
        if (found == std::string::npos)
        {
            return pybind11::cast(std::move(self));
        }

        py_typename.replace(0, strlen("_modmesh.SimpleArray"), "");
        py_typename[0] = tolower(py_typename[0]);
        const DataType dt(py_typename);

#define DECL_MM_TAKE_ALONG_AXIS_TYPED(IntDataType) \
    case DataType::IntDataType:                    \
        return pybind11::cast(self.take_along_axis(indices.cast<SimpleArray##IntDataType>()));

        switch (dt)
        {
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Int8)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Int16)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Int32)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Int64)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Uint8)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Uint16)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Uint32)
            DECL_MM_TAKE_ALONG_AXIS_TYPED(Uint64)
        default:
            break;
        }
        return pybind11::cast(std::move(self));

#undef DECL_MM_TAKE_ALONG_AXIS_TYPED
    }

    static pybind11::object take_along_axis_simd(wrapped_type & self, pybind11::object const & indices)
    {
        std::string py_typename(pybind11::detail::obj_class_name(indices.ptr()));
        const std::size_t found = py_typename.find("_modmesh.SimpleArray");
        if (found == std::string::npos)
        {
            return pybind11::cast(std::move(self));
        }

        py_typename.replace(0, strlen("_modmesh.SimpleArray"), "");
        py_typename[0] = tolower(py_typename[0]);
        const DataType dt(py_typename);

#define DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(IntDataType) \
    case DataType::IntDataType:                         \
        return pybind11::cast(self.take_along_axis_simd(indices.cast<SimpleArray##IntDataType>()));

        switch (dt)
        {
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Int8)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Int16)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Int32)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Int64)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Uint8)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Uint16)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Uint32)
            DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED(Uint64)
        default:
            break;
        }
        return pybind11::cast(std::move(self));

#undef DECL_MM_TAKE_ALONG_AXIS_SIMD_TYPED
    }

    wrapper_type & wrap_search()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def("argmin", &wrapped_type::argmin)
            .def("argmax", &wrapped_type::argmax)
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
                [](size_t length, size_t alignment)
                { return wrapped_type(length, alignment); }),
            py::arg("length"),
            py::arg("alignment") = 0)
        .def_timed(py::init<>())
        .def_timed("reserve", &wrapped_type::reserve, py::arg("cap"))
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        .def_property_readonly("capacity", &wrapped_type::capacity)
        .def_property_readonly("alignment", &wrapped_type::alignment)
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
    WrapSimpleArray<Complex<float>>::commit(mod, "SimpleArrayComplex64", "SimpleArrayComplex64");
    WrapSimpleArray<Complex<double>>::commit(mod, "SimpleArrayComplex128", "SimpleArrayComplex128");

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
    WrapSimpleCollector<Complex<float>>::commit(mod, "SimpleCollectorComplex64", "SimpleCollectorComplex64");
    WrapSimpleCollector<Complex<double>>::commit(mod, "SimpleCollectorComplex128", "SimpleCollectorComplex128");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
