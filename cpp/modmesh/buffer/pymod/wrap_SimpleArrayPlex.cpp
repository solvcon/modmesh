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

namespace modmesh
{

namespace python
{

/// Execute the callback function with the typed array
/// @tparam Callable the type of the callback function
/// @param arrayplex the plex array, which is the wrapper of the typed array
/// @param callback the callback function, which has the typed array as the argument
/// @return the return type of the callback function
template <typename Callable>
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static auto execute_callback_with_typed_array(SimpleArrayPlex & arrayplex, Callable && callback)
{
    switch (arrayplex.data_type())
    {
    case DataType::Bool:
    {
        auto * array = static_cast<SimpleArrayBool *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Int8:
    {
        auto * array = static_cast<SimpleArrayInt8 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Int16:
    {
        auto * array = static_cast<SimpleArrayInt16 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Int32:
    {
        auto * array = static_cast<SimpleArrayInt32 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Int64:
    {
        auto * array = static_cast<SimpleArrayInt64 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Uint8:
    {
        auto * array = static_cast<SimpleArrayUint8 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Uint16:
    {
        auto * array = static_cast<SimpleArrayUint16 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Uint32:
    {
        auto * array = static_cast<SimpleArrayUint32 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Uint64:
    {
        auto * array = static_cast<SimpleArrayUint64 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Float32:
    {
        auto * array = static_cast<SimpleArrayFloat32 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    case DataType::Float64:
    {
        auto * array = static_cast<SimpleArrayFloat64 *>(arrayplex.mutable_instance_ptr());
        return callback(*array);
        break;
    }
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    }
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
                            std::vector<size_t> stride;
                            
                            for (size_t const i : array.stride())
                            {
                                stride.push_back(i * sizeof(data_type));
                            }
                            return pybind11::buffer_info(
                                array.data(), /* Pointer to buffer */
                                sizeof(data_type), /* Size of one scalar */
                                pybind11::format_descriptor<data_type>::format(), /* Python struct-style format descriptor */
                                array.ndim(), /* Number of dimensions */
                                std::vector<size_t>(array.shape().begin(), array.shape().end()), /* Buffer dimensions */
                                stride /* Strides (in bytes) for each index */
                            ); });
                })
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

    /// Return the typed function from the arrayplex
    static pybind11::object
    get_typed_array(wrapped_type const & array_plex)
    {
        switch (array_plex.data_type())
        {
        case DataType::Bool:
        {
            const auto * array = static_cast<const SimpleArrayBool *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayBool(*array)));
        }
        case DataType::Int8:
        {
            const auto * array = static_cast<const SimpleArrayInt8 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayInt8(*array)));
        }
        case DataType::Int16:
        {
            const auto * array = static_cast<const SimpleArrayInt16 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayInt16(*array)));
        }
        case DataType::Int32:
        {
            const auto * array = static_cast<const SimpleArrayInt32 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayInt32(*array)));
        }
        case DataType::Int64:
        {
            const auto * array = static_cast<const SimpleArrayInt64 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayInt64(*array)));
        }
        case DataType::Uint8:
        {
            const auto * array = static_cast<const SimpleArrayUint8 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayUint8(*array)));
        }
        case DataType::Uint16:
        {
            const auto * array = static_cast<const SimpleArrayUint16 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayUint16(*array)));
        }
        case DataType::Uint32:
        {
            const auto * array = static_cast<const SimpleArrayUint32 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayUint32(*array)));
        }
        case DataType::Uint64:
        {
            const auto * array = static_cast<const SimpleArrayUint64 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayUint64(*array)));
        }
        case DataType::Float32:
        {
            const auto * array = static_cast<const SimpleArrayFloat32 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayFloat32(*array)));
        }
        case DataType::Float64:
        {
            const auto * array = static_cast<const SimpleArrayFloat64 *>(array_plex.instance_ptr());
            return pybind11::cast(std::move(SimpleArrayFloat64(*array)));
        }
        default:
        {
            throw std::runtime_error("Unsupported datatype");
        }
        }
    }

    static shape_type make_shape(pybind11::object const & shape_in)
    {
        shape_type shape;
        try
        {
            shape.push_back(shape_in.cast<size_t>());
        }
        catch (const pybind11::cast_error &)
        {
            shape = shape_in.cast<std::vector<size_t>>();
        }
        return shape;
    }

}; /* end of class WrapSimpleArrayPlex*/

void wrap_SimpleArrayPlex(pybind11::module & mod)
{
    WrapSimpleArrayPlex::commit(mod, "SimpleArray", "SimpleArray");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
