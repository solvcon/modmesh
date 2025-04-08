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
#include <modmesh/math/math.hpp>

#include <unordered_map>

namespace modmesh
{

namespace detail
{

#if defined(__aarch64__)
#define vec_typ(typ, N) typ##x##N##_t
#define t_typ(typ) typ##_t

#define DECL_MM_IMPL_CHECK_IDX_RNG_NEON(typ, N_vec, typ_symbol)                                               \
    template <>                                                                                               \
    t_typ(typ) const * check_index_range<t_typ(typ)>(SimpleArray<t_typ(typ)> const & indices, size_t max_idx) \
    {                                                                                                         \
        check_type_range(t_typ(typ), max_idx);                                                                \
                                                                                                              \
        vec_typ(typ, N_vec) max_vec = vdupq_n_##typ_symbol(static_cast<t_typ(typ)>(max_idx));                 \
        vec_typ(typ, N_vec) data_vec = {};                                                                    \
        vec_typ(typ, N_vec) cmp_vec = {};                                                                     \
                                                                                                              \
        t_typ(typ) const * src = indices.begin();                                                             \
        t_typ(typ) const * const end = indices.end();                                                         \
                                                                                                              \
        for (; src <= end - N_vec; src += N_vec)                                                              \
        {                                                                                                     \
            data_vec = vld1q_##typ_symbol(src);                                                               \
            cmp_vec = vcgeq_##typ_symbol(data_vec, max_vec);                                                  \
            if (vgetq_lane_##typ_symbol(cmp_vec, 0) ||                                                        \
                vgetq_lane_##typ_symbol(cmp_vec, 1))                                                          \
            {                                                                                                 \
                goto OUT_OF_RANGE;                                                                            \
            }                                                                                                 \
        }                                                                                                     \
                                                                                                              \
        while (src < end)                                                                                     \
        {                                                                                                     \
            t_typ(typ) idx = *src;                                                                            \
            if (idx < 0 || static_cast<size_t>(idx) > max_idx)                                                \
            {                                                                                                 \
                return src;                                                                                   \
            }                                                                                                 \
            ++src;                                                                                            \
        }                                                                                                     \
        return nullptr;                                                                                       \
                                                                                                              \
    OUT_OF_RANGE:                                                                                             \
        constexpr size_t N = 16;                                                                              \
        t_typ(typ) cmp_val[N] = {};                                                                           \
        t_typ(typ) * cmp = cmp_val;                                                                           \
        vst1q_##typ_symbol(cmp_val, cmp_vec);                                                                 \
                                                                                                              \
        for (size_t i = 0; i < N; ++i, ++cmp)                                                                 \
        {                                                                                                     \
            if (*cmp)                                                                                         \
            {                                                                                                 \
                return src + i;                                                                               \
            }                                                                                                 \
        }                                                                                                     \
        return src;                                                                                           \
    }

DECL_MM_IMPL_CHECK_IDX_RNG_NEON(uint8, 16, u8)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(uint16, 8, u16)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(uint32, 4, u32)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(uint64, 2, u64)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(int8, 16, s8)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(int16, 8, s16)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(int32, 4, s32)
DECL_MM_IMPL_CHECK_IDX_RNG_NEON(int64, 2, s64)

#undef DECL_MM_IMPL_CHECK_IDX_RNG_NEON
#undef t_typ
#undef vec_typ

#endif /* defined(__aarch64__) */

struct DataTypeHasher
{
    // The std::hash<std::string> is not deterministic, so we implement our own.
    // We can use the first and last character to hash the string.
    // The first character indicates the type of the data, and the last character indicates the size of the data (except for bool, which is a special case).
    // The combination of the first and last character can uniquely identify the data type.
    // e.g. "int32"   -> 'i' << 8 | '2' = 26930
    //      "uint32"  -> 'u' << 8 | '2' = 30002
    //      "float64" -> 'f' << 8 | '4' = 26164
    std::size_t operator()(const std::string & data_type_string) const
    {
        assert(data_type_string.size() >= 4); // the length of "bool" and "int8" is 4, and all other data types are longer than 4
        const char ch1 = data_type_string.front();
        const char ch2 = data_type_string.back();
        return static_cast<std::size_t>(ch1) << 8 | static_cast<std::size_t>(ch2);
    }
}; /* end struct DataTypeHasher */

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
    {"float64", DataType::Float64},
    {"complex64", DataType::Complex64},
    {"complex128", DataType::Complex128}};

} /* end namespace detail */

DataType::DataType(const std::string & data_type_string)
{
    auto it = detail::string_data_type_map.find(data_type_string);
    if (it == detail::string_data_type_map.end())
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    m_data_type = it->second;
}

template <>
DataType DataType::from<bool>()
{
    return DataType::Bool;
}

template <>
DataType DataType::from<int8_t>()
{
    return DataType::Int8;
}

template <>
DataType DataType::from<int16_t>()
{
    return DataType::Int16;
}

template <>
DataType DataType::from<int32_t>()
{
    return DataType::Int32;
}

template <>
DataType DataType::from<int64_t>()
{
    return DataType::Int64;
}

template <>
DataType DataType::from<uint8_t>()
{
    return DataType::Uint8;
}

template <>
DataType DataType::from<uint16_t>()
{
    return DataType::Uint16;
}

template <>
DataType DataType::from<uint32_t>()
{
    return DataType::Uint32;
}

template <>
DataType DataType::from<uint64_t>()
{
    return DataType::Uint64;
}

template <>
DataType DataType::from<float>()
{
    return DataType::Float32;
}

template <>
DataType DataType::from<double>()
{
    return DataType::Float64;
}

template <>
DataType DataType::from<Complex<float>>()
{
    return DataType::Complex64;
}

template <>
DataType DataType::from<Complex<double>>()
{
    return DataType::Complex128;
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
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex64, SimpleArrayComplex64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex128, SimpleArrayComplex128, shape)
    default:
        throw std::invalid_argument("Unsupported datatype");
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
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex64, SimpleArrayComplex64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex128, SimpleArrayComplex128, shape, buffer)
    default:
        throw std::invalid_argument("Unsupported datatype");
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
    case DataType::Complex64:
    {
        const auto * array = static_cast<SimpleArrayComplex64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex64(*array));
        break;
    }
    case DataType::Complex128:
    {
        const auto * array = static_cast<SimpleArrayComplex128 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex128(*array));
        break;
    }
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
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
    case DataType::Complex64:
    {
        const auto * array = static_cast<SimpleArrayComplex64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex64(*array));
        break;
    }
    case DataType::Complex128:
    {
        const auto * array = static_cast<SimpleArrayComplex128 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex128(*array));
        break;
    }
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
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
    case DataType::Complex64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayComplex64 *>(m_instance_ptr);
        break;
    }
    case DataType::Complex128:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayComplex128 *>(m_instance_ptr);
        break;
    }
    default:
        break;
    }

    m_instance_ptr = nullptr;
    m_has_instance_ownership = false;
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
