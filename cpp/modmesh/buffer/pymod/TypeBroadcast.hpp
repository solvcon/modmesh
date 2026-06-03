#pragma once

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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h> // Must be the first include.

namespace modmesh
{
namespace python
{

namespace detail
{

inline modmesh::detail::shape_type shape_from_slices(
    std::vector<modmesh::detail::slice_type> const & slices)
{
    modmesh::detail::shape_type shape(slices.size());
    for (size_t i = 0; i < slices.size(); ++i)
    {
        shape[i] = static_cast<size_t>(slices[i][3]);
    }
    return shape;
}

} /* end namespace detail */

template <typename T /* original type */, typename D /* for destination type */>
struct TypeBroadcastImpl
{
    using slice_type = modmesh::detail::slice_type;
    using shape_type = modmesh::detail::shape_type;

    static ssize_t input_offset(pybind11::array_t<D> const & arr_in, shape_type const & sidx)
    {
        ssize_t offset = 0;
        for (pybind11::ssize_t i = 0; i < arr_in.ndim(); ++i)
        {
            auto const index = static_cast<ssize_t>(sidx[i]);
            offset += arr_in.strides(i) / arr_in.itemsize() * index;
        }
        return offset;
    }

    static ssize_t offset_from_slices(SimpleArray<T> const & arr, std::vector<slice_type> const & slices, shape_type const & sidx)
    {
        ssize_t offset = 0;
        for (size_t i = 0; i < arr.ndim(); ++i)
        {
            auto const slice_index = static_cast<ssize_t>(sidx[i]);
            ssize_t const index = slices[i][0] + slice_index * slices[i][2];
            offset += modmesh::detail::stride_to_signed(arr.stride(i)) * index;
        }
        return offset;
    }

    // NOLINTNEXTLINE(misc-no-recursion)
    static void copy_idx(SimpleArray<T> & arr_out, std::vector<slice_type> const & slices, pybind11::array_t<D> const * arr_in, shape_type left_shape, shape_type sidx, int dim)
    {
        using out_type = typename std::remove_reference_t<decltype(arr_out[0])>;

        if (dim < 0)
        {
            D const * ptr_in = arr_in->data() + input_offset(*arr_in, sidx);
            ssize_t const offset_out = offset_from_slices(arr_out, slices, sidx);

            constexpr bool valid_conversion = (!is_complex_v<T> && !is_complex_v<D>) || (is_complex_v<T> && is_complex_v<D> && std::is_same_v<T, D>);

            if constexpr (valid_conversion)
            {
                // FIXME: NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
                arr_out.data()[offset_out] = static_cast<out_type>(*ptr_in);
            }
            else
            {
                throw std::runtime_error("Cannot convert between complex and non-complex types");
            }
            return;
        }

        for (size_t i = 0; i < left_shape[dim]; ++i)
        {
            sidx[dim] = i;
            copy_idx(arr_out, slices, arr_in, left_shape, sidx, dim - 1);
        }
    }

    static void broadcast(SimpleArray<T> & arr_out, std::vector<slice_type> const & slices, pybind11::array const & arr_in)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto * arr_new = reinterpret_cast<pybind11::array_t<D> const *>(&arr_in);

        shape_type const left_shape = modmesh::python::detail::shape_from_slices(slices);
        shape_type const sidx_init(arr_out.ndim());
        copy_idx(arr_out, slices, arr_new, left_shape, sidx_init, static_cast<int>(arr_out.ndim()) - 1);
    }
}; /* end struct TypeBroadcastImpl */

template <typename T>
struct TypeBroadcast
{
    using slice_type = modmesh::detail::slice_type;
    using shape_type = modmesh::detail::shape_type;

    static void check_shape(SimpleArray<T> const & arr_out, std::vector<slice_type> const & slices, pybind11::array const & arr_in)
    {
        shape_type right_shape(arr_in.ndim());
        for (pybind11::ssize_t i = 0; i < arr_in.ndim(); i++)
        {
            right_shape[i] = arr_in.shape(i);
        }

        shape_type left_shape = modmesh::python::detail::shape_from_slices(slices);

        if (arr_out.ndim() != static_cast<size_t>(arr_in.ndim()))
        {
            throw_shape_error(left_shape, right_shape);
        }

        for (size_t i = 0; i < left_shape.size(); ++i)
        {
            if (left_shape[i] != static_cast<size_t>(right_shape[i]))
            {
                throw_shape_error(left_shape, right_shape);
            }
        }
    }

    static void broadcast(SimpleArray<T> & arr_out, std::vector<slice_type> const & slices, pybind11::array const & arr_in)
    {
        if (dtype_is_type<bool>(arr_in))
        {
            TypeBroadcastImpl<T, bool>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int8_t>(arr_in))
        {
            TypeBroadcastImpl<T, int8_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int16_t>(arr_in))
        {
            TypeBroadcastImpl<T, int16_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int32_t>(arr_in))
        {
            TypeBroadcastImpl<T, int32_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<int64_t>(arr_in))
        {
            TypeBroadcastImpl<T, int64_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint8_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint8_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint16_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint16_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint32_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint32_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<uint64_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint64_t>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<float>(arr_in))
        {
            TypeBroadcastImpl<T, float>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<double>(arr_in))
        {
            TypeBroadcastImpl<T, double>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<Complex<float>>(arr_in))
        {
            TypeBroadcastImpl<T, Complex<float>>::broadcast(arr_out, slices, arr_in);
        }
        else if (dtype_is_type<Complex<double>>(arr_in))
        {
            TypeBroadcastImpl<T, Complex<double>>::broadcast(arr_out, slices, arr_in);
        }
        else
        {
            throw std::runtime_error("input array data type not support!");
        }
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    static void throw_shape_error(shape_type const & left_shape, shape_type const & right_shape)
    {

        std::ostringstream msg;
        msg << "Broadcast input array from shape(";
        for (size_t i = 0; i < right_shape.size(); ++i)
        {
            msg << right_shape[i];
            if (i != right_shape.size() - 1)
            {
                msg << ", ";
            }
        }
        msg << ") into shape(";
        for (size_t i = 0; i < left_shape.size(); ++i)
        {
            msg << left_shape[i];
            if (i != left_shape.size() - 1)
            {
                msg << ", ";
            }
        }
        msg << ")";

        throw std::runtime_error(msg.str());
    }
}; /* end struct TypeBroadCast */

} /* end namespace python */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
