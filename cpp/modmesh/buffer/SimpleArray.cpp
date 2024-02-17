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

namespace modmesh
{

DataType get_data_type_from_string(const std::string & data_type_string)
{
    if (data_type_string == "bool")
    {
        return DataType::Bool;
    }
    if (data_type_string == "int8")
    {
        return DataType::Int8;
    }
    if (data_type_string == "int16")
    {
        return DataType::Int16;
    }
    if (data_type_string == "int32")
    {
        return DataType::Int32;
    }
    if (data_type_string == "int64")
    {
        return DataType::Uint64;
    }
    if (data_type_string == "uint8")
    {
        return DataType::Uint8;
    }
    if (data_type_string == "uint16")
    {
        return DataType::Uint16;
    }
    if (data_type_string == "uint32")
    {
        return DataType::Uint32;
    }
    if (data_type_string == "uint64")
    {
        return DataType::Uint64;
    }
    if (data_type_string == "float32")
    {
        return DataType::Float32;
    }
    if (data_type_string == "float64")
    {
        return DataType::Float64;
    }
    throw std::runtime_error("Unsupported datatype");
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

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, DataType data_type)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
    switch (data_type)
    {
    case DataType::Bool:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayBool(shape));
        break;
    }
    case DataType::Int8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt8(shape));
        break;
    }
    case DataType::Int16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt16(shape));
        break;
    }
    case DataType::Int32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt32(shape));
        break;
    }
    case DataType::Int64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt64(shape));
        break;
    }
    case DataType::Uint8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint8(shape));
        break;
    }
    case DataType::Uint16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint16(shape));
        break;
    }
    case DataType::Uint32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint32(shape));
        break;
    }
    case DataType::Uint64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint64(shape));
        break;
    }
    case DataType::Float32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat32(shape));
        break;
    }
    case DataType::Float64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat64(shape));
        break;
    }
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
}

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
