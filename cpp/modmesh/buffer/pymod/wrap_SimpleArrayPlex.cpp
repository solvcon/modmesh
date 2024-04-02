/*
 * Copyright (c) 2024, An-Chi Liu <phy.tiger@gmail.com>
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
#include <type_traits>

namespace modmesh
{

namespace python
{

/// Execute the callback function with the typed array
/// @tparam A either `SimpleArrayPlex` or `const SimpleArrayPlex`
/// @tparam C the type of the callback function
/// @param arrayplex the plex array, which is the wrapper of the typed array
/// @param callback the callback function, which has the mutable typed array as the argument
/// @return the return type of the callback function
template <typename A, typename C, typename = std::enable_if_t<std::is_same_v<std::remove_const_t<A>, SimpleArrayPlex>>>
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static auto execute_callback_with_typed_array(A & arrayplex, C && callback)
{
// We get the typed array from the arrayplex and call the callback function with the typed array.
#define DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType, ArrayType)                                                     \
    case DataType:                                                                                                     \
    {                                                                                                                  \
        using ArrayTypePtr = typename std::conditional<std::is_const<A>::value, const ArrayType *, ArrayType *>::type; \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */                                              \
        auto array = reinterpret_cast<ArrayTypePtr>(arrayplex.mutable_instance_ptr());                                 \
        return callback(*array);                                                                                       \
    }

    switch (arrayplex.data_type())
    {
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Bool, SimpleArrayBool)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int8, SimpleArrayInt8)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int16, SimpleArrayInt16)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int32, SimpleArrayInt32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int64, SimpleArrayInt64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Float64, SimpleArrayFloat64)
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    }

#undef DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY
}

/// Check the data type of the python value match the given data type. If not, throw a type error.
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static void verify_python_value_datatype(pybind11::object const & value, DataType datatype)
{
    switch (datatype)
    {
    case DataType::Bool:
    {
        if (!pybind11::isinstance<pybind11::bool_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python bool");
        }
        break;
    }
    case DataType::Int8:
    case DataType::Int16:
    case DataType::Int32:
    case DataType::Int64:
    case DataType::Uint8:
    case DataType::Uint16:
    case DataType::Uint32:
    case DataType::Uint64:
    {
        if (!pybind11::isinstance<pybind11::int_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python int");
        }
        break;
    }
    case DataType::Float32:
    case DataType::Float64:
    {
        if (!pybind11::isinstance<pybind11::float_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python float");
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported datatype");
    }
}

/// Get the typed array value by the key
/// @tparam T the type of the key
template <typename T>
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static pybind11::object get_typed_array_value(const SimpleArrayPlex & array_plex, T key)
{
#define DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType, ArrayType)                     \
    case DataType:                                                                      \
    {                                                                                   \
        const auto * array = static_cast<const ArrayType *>(array_plex.instance_ptr()); \
        return pybind11::cast(array->at(key));                                          \
    }

    switch (array_plex.data_type())
    {
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Bool, SimpleArrayBool)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int8, SimpleArrayInt8)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int16, SimpleArrayInt16)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int32, SimpleArrayInt32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int64, SimpleArrayInt64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Float64, SimpleArrayFloat64)
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
#undef DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX
}

/// Get the typed array from the arrayplex
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static pybind11::object get_typed_array(const SimpleArrayPlex & array_plex)
{
#define DECL_MM_GET_TYPED_ARRAY(DataType, ArrayType)                                    \
    case DataType:                                                                      \
    {                                                                                   \
        const auto * array = static_cast<const ArrayType *>(array_plex.instance_ptr()); \
        return pybind11::cast(std::move(ArrayType(*array)));                            \
    }

    switch (array_plex.data_type())
    {
        DECL_MM_GET_TYPED_ARRAY(DataType::Bool, SimpleArrayBool)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int8, SimpleArrayInt8)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int16, SimpleArrayInt16)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int32, SimpleArrayInt32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int64, SimpleArrayInt64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Float64, SimpleArrayFloat64)
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
#undef DECL_MM_GET_TYPED_ARRAY
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapSimpleArrayPlex : public WrapBase<WrapSimpleArrayPlex, SimpleArrayPlex>
{
    using root_base_type = WrapBase<WrapSimpleArrayPlex, SimpleArrayPlex>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using wrapper_type = typename root_base_type::wrapper_type;
    using shape_type = modmesh::detail::shape_type;

    friend root_base_type;

    WrapSimpleArrayPlex(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
    {
        (*this)
            .def_timed(
                pybind11::init(
                    [](pybind11::object const & shape, std::string const & datatype)
                    { return wrapped_type(make_shape(shape), datatype); }),
                pybind11::arg("shape"),
                pybind11::arg("dtype"))
            .def_timed(
                pybind11::init(&init_array_plex_with_value),
                pybind11::arg("shape"),
                pybind11::arg("value"),
                pybind11::arg("dtype"))
            .def(
                pybind11::init(
                    [](pybind11::array & arr_in)
                    {
                        shape_type shape;
                        for (ssize_t i = 0; i < arr_in.ndim(); ++i)
                        {
                            shape.push_back(arr_in.shape(i));
                        }
                        std::shared_ptr<ConcreteBuffer> const buffer = ConcreteBuffer::construct(
                            arr_in.nbytes(),
                            arr_in.mutable_data(),
                            std::make_unique<ConcreteBufferNdarrayRemover>(arr_in));
                        return wrapped_type(shape, buffer, pybind11::str(arr_in.dtype()));
                    }),
                pybind11::arg("array"))
            .def_property_readonly("typed", &get_typed_array)
            .def_buffer(
                [](wrapped_type & self)
                {
                    return execute_callback_with_typed_array(
                        self,
                        [](auto & array)
                        {
                            using data_type = typename std::remove_reference_t<decltype(array[0])>;
                            return get_buffer_info<data_type>(array); });
                })
            .def_property_readonly("nbytes", [](wrapped_type & self)
                                   { return execute_callback_with_typed_array(
                                         self, [](auto & array)
                                         { return array.nbytes(); }); })
            .def_property_readonly("size", [](wrapped_type & self)
                                   { return execute_callback_with_typed_array(
                                         self, [](auto & array)
                                         { return array.size(); }); })
            .def_property_readonly("itemsize", [](wrapped_type & self)
                                   { return execute_callback_with_typed_array(
                                         self, [](auto & array)
                                         { return array.itemsize(); }); })
            .def_property_readonly(
                "shape",
                [](wrapped_type const & self)
                {
                    return execute_callback_with_typed_array(
                        self, [](const auto & array)
                        {
                            using data_type = typename std::remove_const_t<std::remove_reference_t<decltype(array[0])>>;
                            return get_array_shape<data_type>(array); });
                })
            .def_property_readonly(
                "stride",
                [](wrapped_type const & self)
                {
                    return execute_callback_with_typed_array(
                        self, [](const auto & array)
                        {
                            using data_type = typename std::remove_const_t<std::remove_reference_t<decltype(array[0])>>;
                            return get_array_stride<data_type>(array); });
                })
            .def("__len__", [](wrapped_type & self)
                 { return execute_callback_with_typed_array(
                       self, [](auto & array)
                       { return array.size(); }); })
            .def("__getitem__", &get_typed_array_value<ssize_t>)
            .def("__getitem__", &get_typed_array_value<const std::vector<ssize_t> &>)
            .def("__setitem__", [](wrapped_type & self, pybind11::args const & args)
                 { return execute_callback_with_typed_array(
                       self, [&args](auto & array)
                       { 
                            using data_type = typename std::remove_reference_t<decltype(array[0])>;
                            setitem_parser<data_type>(array, args); }); })
            .def("reshape", [](wrapped_type const & self, pybind11::object const & py_shape)
                 { return execute_callback_with_typed_array(
                       self,
                       // NOLINTNEXTLINE(fuchsia-trailing-return)
                       [&](const auto & array) -> wrapped_type // need the return type to get correct deduced type
                       { 
                        const auto shape = make_shape(py_shape);
                        return array.reshape(shape); }); })
            .def_property_readonly("has_ghost", [](wrapped_type & self)
                                   { return execute_callback_with_typed_array(
                                         self, [](auto & array)
                                         { return array.has_ghost(); }); })
            .def_property_readonly("nbody", [](wrapped_type & self)
                                   { return execute_callback_with_typed_array(
                                         self, [](auto & array)
                                         { return array.nbody(); }); })
            /// TODO: should have the same interface as WrapSimpleArray
            ;
    }

    /// Initialize the arrayplex with the given value
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    static wrapped_type init_array_plex_with_value(pybind11::object const & shape_in, pybind11::object const & py_value, std::string const & datatype_str)
    {
        const shape_type shape = make_shape(shape_in);
        wrapped_type array_plex(shape, datatype_str);
        auto datatype = array_plex.data_type();
        execute_callback_with_typed_array(
            array_plex, [&py_value, datatype](auto & array)
            { 
                using value_type = typename std::remove_reference_t<decltype(array[0])>;
                verify_python_value_datatype(py_value, datatype);
                const auto value = py_value.cast<value_type>();
                array.fill(value); });
        return array_plex;
    }

}; /* end of class WrapSimpleArrayPlex*/

void wrap_SimpleArrayPlex(pybind11::module & mod)
{
    WrapSimpleArrayPlex::commit(mod, "SimpleArray", "SimpleArray");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
