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

#include <modmesh/buffer/SimpleArray.hpp>

#include <unordered_map>

namespace modmesh
{

namespace detail
{
struct DataTypeHasher
{
    std::size_t operator()(const std::string & data_type_string) const
    {
        const char ch1 = data_type_string.front();
        const char ch2 = data_type_string.back();
        return static_cast<std::size_t>(ch1) << 8 | static_cast<std::size_t>(ch2);
    }
};

// NOLINTNEXTLINE(misc-use-anonymous-namespace, cert-err58-cpp, cppcoreguidelines-avoid-non-const-global-variables, fuchsia-statically-constructed-objects)
static std::unordered_map<std::string, DataType, DataTypeHasher> string_data_type_map = {
    {"bool", DataType::Bool},
    {"int8", DataType::Int8},
    {"int16", DataType::Int16},
    {"int32", DataType::Int32},
    {"int64", DataType::Int64},
    {"uint8", DataType::Uint8},
    {"uint16", DataType::Uint16},
    {"uint32", DataType::Uint32},
    {"uint64", DataType::Uint64},
    {"float32", DataType::Float32},
    {"float64", DataType::Float64}};
} // end of namespace detail

DataType::DataType(const std::string & data_type_string)
{
    auto it = detail::string_data_type_map.find(data_type_string);
    if (it == detail::string_data_type_map.end())
    {
        throw std::runtime_error("Unsupported datatype");
    }
    m_data_type = it->second;
}

template <>
DataType get_data_type_from_type<bool>()
{
    return DataType::Bool;
}

template <>
DataType get_data_type_from_type<int8_t>()
{
    return DataType::Int8;
}

template <>
DataType get_data_type_from_type<int16_t>()
{
    return DataType::Int16;
}

template <>
DataType get_data_type_from_type<int32_t>()
{
    return DataType::Int32;
}

template <>
DataType get_data_type_from_type<int64_t>()
{
    return DataType::Int64;
}

template <>
DataType get_data_type_from_type<uint8_t>()
{
    return DataType::Uint8;
}

template <>
DataType get_data_type_from_type<uint16_t>()
{
    return DataType::Uint16;
}

template <>
DataType get_data_type_from_type<uint32_t>()
{
    return DataType::Uint32;
}

template <>
DataType get_data_type_from_type<uint64_t>()
{
    return DataType::Uint64;
}

template <>
DataType get_data_type_from_type<float>()
{
    return DataType::Float32;
}

template <>
DataType get_data_type_from_type<double>()
{
    return DataType::Float64;
}

// According to the `DataType`, create the corresponding `SimpleArray<T>` instance
// and assign it to `m_instance_ptr`. The `m_instance_ptr` is a void pointer, so
// we need to use `reinterpret_cast` to convert the pointer of the array instance.
#define DECL_MM_CREATE_SIMPLE_ARRAY(DataType, ArrayType, ...)                  \
    case DataType:                                                             \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */      \
        m_instance_ptr = reinterpret_cast<void *>(new ArrayType(__VA_ARGS__)); \
        break;

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, const DataType data_type)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
    switch (data_type)
    {
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Bool, SimpleArrayBool, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int8, SimpleArrayInt8, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int16, SimpleArrayInt16, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int32, SimpleArrayInt32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int64, SimpleArrayInt64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint8, SimpleArrayUint8, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint16, SimpleArrayUint16, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint32, SimpleArrayUint32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint64, SimpleArrayUint64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float32, SimpleArrayFloat32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float64, SimpleArrayFloat64, shape)
    default:
        throw std::runtime_error("Unsupported datatype");
    }
}

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, const DataType data_type)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
    switch (data_type)
    {
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Bool, SimpleArrayBool, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int8, SimpleArrayInt8, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int16, SimpleArrayInt16, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int32, SimpleArrayInt32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int64, SimpleArrayInt64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint8, SimpleArrayUint8, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint16, SimpleArrayUint16, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint32, SimpleArrayUint32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint64, SimpleArrayUint64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float32, SimpleArrayFloat32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float64, SimpleArrayFloat64, shape, buffer)
    default:
        throw std::runtime_error("Unsupported datatype");
    }
}

#undef DECL_MM_CREATE_SIMPLE_ARRAY

SimpleArrayPlex::SimpleArrayPlex(SimpleArrayPlex const & other)
    : m_data_type(other.m_data_type)
{
    if (!other.m_instance_ptr)
    {
        return; // other does not have instance
    }

    m_has_instance_ownership = true;

    switch (other.m_data_type)
    {
    case DataType::Bool:
    {
        const auto * array = static_cast<SimpleArrayBool *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayBool(*array));
        break;
    }
    case DataType::Int8:
    {
        const auto * array = static_cast<SimpleArrayInt8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt8(*array));
        break;
    }
    case DataType::Int16:
    {
        const auto * array = static_cast<SimpleArrayInt16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt16(*array));
        break;
    }
    case DataType::Int32:
    {
        const auto * array = static_cast<SimpleArrayInt32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt32(*array));
        break;
    }
    case DataType::Int64:
    {
        const auto * array = static_cast<SimpleArrayInt64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt64(*array));
        break;
    }
    case DataType::Uint8:
    {
        const auto * array = static_cast<SimpleArrayUint8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint8(*array));
        break;
    }
    case DataType::Uint16:
    {
        const auto * array = static_cast<SimpleArrayUint16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint16(*array));
        break;
    }
    case DataType::Uint32:
    {
        const auto * array = static_cast<SimpleArrayUint32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint32(*array));
        break;
    }
    case DataType::Uint64:
    {
        const auto * array = static_cast<SimpleArrayUint64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint64(*array));
        break;
    }
    case DataType::Float32:
    {
        const auto * array = static_cast<SimpleArrayFloat32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat32(*array));
        break;
    }
    case DataType::Float64:
    {
        const auto * array = static_cast<SimpleArrayFloat64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat64(*array));
        break;
    }
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
}

SimpleArrayPlex::SimpleArrayPlex(SimpleArrayPlex && other)
    : m_data_type(other.m_data_type)
{
    if (!other.m_instance_ptr)
    {
        return; // other does not have instance
    }

    // take onwership
    m_has_instance_ownership = true;
    other.m_has_instance_ownership = false;

    m_instance_ptr = other.m_instance_ptr;
}

SimpleArrayPlex & SimpleArrayPlex::operator=(SimpleArrayPlex const & other)
{
    if (this == &other)
    {
        return *this;
    }

    m_data_type = other.m_data_type;

    if (!other.m_instance_ptr)
    {
        return *this; // other does not have instance
    }

    m_has_instance_ownership = true;

    switch (other.m_data_type)
    {
    case DataType::Bool:
    {
        const auto * array = static_cast<SimpleArrayBool *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayBool(*array));
        break;
    }
    case DataType::Int8:
    {
        const auto * array = static_cast<SimpleArrayInt8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt8(*array));
        break;
    }
    case DataType::Int16:
    {
        const auto * array = static_cast<SimpleArrayInt16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt16(*array));
        break;
    }
    case DataType::Int32:
    {
        const auto * array = static_cast<SimpleArrayInt32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt32(*array));
        break;
    }
    case DataType::Int64:
    {
        const auto * array = static_cast<SimpleArrayInt64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt64(*array));
        break;
    }
    case DataType::Uint8:
    {
        const auto * array = static_cast<SimpleArrayUint8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint8(*array));
        break;
    }
    case DataType::Uint16:
    {
        const auto * array = static_cast<SimpleArrayUint16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint16(*array));
        break;
    }
    case DataType::Uint32:
    {
        const auto * array = static_cast<SimpleArrayUint32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint32(*array));
        break;
    }
    case DataType::Uint64:
    {
        const auto * array = static_cast<SimpleArrayUint64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint64(*array));
        break;
    }
    case DataType::Float32:
    {
        const auto * array = static_cast<SimpleArrayFloat32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat32(*array));
        break;
    }
    case DataType::Float64:
    {
        const auto * array = static_cast<SimpleArrayFloat64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat64(*array));
        break;
    }
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
    return *this;
}

SimpleArrayPlex & SimpleArrayPlex::operator=(SimpleArrayPlex && other)
{
    m_data_type = other.m_data_type;

    if (!other.m_instance_ptr)
    {
        return *this; // other does not have instance
    }

    // take onwership
    m_has_instance_ownership = true;
    other.m_has_instance_ownership = false;

    m_instance_ptr = other.m_instance_ptr;
    return *this;
}

SimpleArrayPlex::~SimpleArrayPlex()
{
    if (m_instance_ptr == nullptr || !m_has_instance_ownership)
    {
        return;
    }

    switch (m_data_type)
    {
    case DataType::Bool:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayBool *>(m_instance_ptr);
        break;
    }
    case DataType::Int8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt8 *>(m_instance_ptr);
        break;
    }
    case DataType::Int16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt16 *>(m_instance_ptr);
        break;
    }
    case DataType::Int32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt32 *>(m_instance_ptr);
        break;
    }
    case DataType::Int64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt64 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint8 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint16 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint32 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint64 *>(m_instance_ptr);
        break;
    }
    case DataType::Float32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayFloat32 *>(m_instance_ptr);
        break;
    }
    case DataType::Float64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayFloat64 *>(m_instance_ptr);
        break;
    }
    default:
        break;
    }

    m_instance_ptr = nullptr;
    m_has_instance_ownership = false;
}

} /* end namespace modmesh */
