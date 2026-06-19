#pragma once
/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <modmesh/buffer/buffer.hpp>
#include <modmesh/math/math.hpp>

/**
 * The purpose of including this header is to facilitate implicit casting of
 * SimpleArray types, such as casting from SimpleArrayPlex to SimplyArray.
 * It should be incorporated whenever there is a C++ function wrapped by
 * Pybind11 that involves SimpleArray as either parameters or a return value.
 */

namespace pybind11
{
namespace detail
{

// Define the Pybind11 caster for SimpleArray<T>
#define DECL_MM_SIMPLE_ARRAY_CASTER(DATATYPE)                                                                                                                 \
    template <> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                                                                              \
    struct type_caster<modmesh::SimpleArray##DATATYPE> : public type_caster_base<modmesh::SimpleArray##DATATYPE>                                              \
    {                                                                                                                                                         \
        using base = type_caster_base<modmesh::SimpleArray##DATATYPE>;                                                                                        \
                                                                                                                                                              \
    public:                                                                                                                                                   \
        /* Conversion from Python object to C++ */                                                                                                            \
        bool load(pybind11::handle src, bool convert)                                                                                                         \
        {                                                                                                                                                     \
            /* Check if the source is SimpleArray */                                                                                                          \
            if (base::load(src, convert))                                                                                                                     \
            {                                                                                                                                                 \
                return true;                                                                                                                                  \
            }                                                                                                                                                 \
            /* Check if the source object is a valid SimpleArrayPlex  */                                                                                      \
            if (!pybind11::isinstance<modmesh::SimpleArrayPlex>(src))                                                                                         \
            {                                                                                                                                                 \
                return false;                                                                                                                                 \
            }                                                                                                                                                 \
                                                                                                                                                              \
            /* Get the SimpleArrayPlex object from the source handle */                                                                                       \
            modmesh::SimpleArrayPlex arrayplex = src.cast<modmesh::SimpleArrayPlex>(); /* NOLINT(misc-const-correctness) */                                   \
                                                                                                                                                              \
            /* Check if the data type is matched */                                                                                                           \
            if (arrayplex.data_type() != modmesh::DataType::DATATYPE)                                                                                         \
            {                                                                                                                                                 \
                return false;                                                                                                                                 \
            }                                                                                                                                                 \
                                                                                                                                                              \
            /* construct the new array from the arrayplex NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */                                      \
            const modmesh::SimpleArray##DATATYPE * array_from_arrayplex = reinterpret_cast<const modmesh::SimpleArray##DATATYPE *>(arrayplex.instance_ptr()); \
            value = const_cast<modmesh::SimpleArray##DATATYPE *>(array_from_arrayplex); /* NOLINT(cppcoreguidelines-pro-type-const-cast) */                   \
            return true;                                                                                                                                      \
        }                                                                                                                                                     \
                                                                                                                                                              \
        /* Conversion from C++ to Python object FIXME: NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved) */                                  \
        static pybind11::handle cast(modmesh::SimpleArray##DATATYPE && src, pybind11::return_value_policy policy, pybind11::handle parent)                    \
        {                                                                                                                                                     \
            return base::cast(src, policy, parent);                                                                                                           \
        }                                                                                                                                                     \
    }

DECL_MM_SIMPLE_ARRAY_CASTER(Bool);
DECL_MM_SIMPLE_ARRAY_CASTER(Int8);
DECL_MM_SIMPLE_ARRAY_CASTER(Int16);
DECL_MM_SIMPLE_ARRAY_CASTER(Int32);
DECL_MM_SIMPLE_ARRAY_CASTER(Int64);
DECL_MM_SIMPLE_ARRAY_CASTER(Uint8);
DECL_MM_SIMPLE_ARRAY_CASTER(Uint16);
DECL_MM_SIMPLE_ARRAY_CASTER(Uint32);
DECL_MM_SIMPLE_ARRAY_CASTER(Uint64);
DECL_MM_SIMPLE_ARRAY_CASTER(Float32);
DECL_MM_SIMPLE_ARRAY_CASTER(Float64);
DECL_MM_SIMPLE_ARRAY_CASTER(Complex64);
DECL_MM_SIMPLE_ARRAY_CASTER(Complex128);

#undef DECL_MM_SIMPLE_ARRAY_CASTER

} /* end namespace detail */
} /* end namespace pybind11 */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
