#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <modmesh/buffer/SimpleArray.hpp>

namespace modmesh
{
namespace python
{

template <typename T /* original type */, typename D /* for destination type */>
struct TypeBroadcastImpl
{
    using slice_type = small_vector<int>;
    using shape_type = typename SimpleArray<T>::shape_type;

    // NOLINTNEXTLINE(misc-no-recursion)
    static void copy_idx(SimpleArray<T> & arr_out, std::vector<slice_type> const & slices, pybind11::array_t<D> const * arr_in, shape_type left_shape, shape_type sidx, int dim)
    {
        using out_type = typename std::remove_reference_t<decltype(arr_out[0])>;

        if (dim < 0)
        {
            return;
        }

        for (size_t i = 0; i < left_shape[dim]; ++i)
        {
            sidx[dim] = i;

            size_t offset_in = 0;
            for (pybind11::ssize_t it = 0; it < arr_in->ndim(); ++it)
            {
                offset_in += arr_in->strides(it) / arr_in->itemsize() * sidx[it];
            }
            const D * ptr_in = arr_in->data() + offset_in;

            size_t offset_out = 0;
            for (size_t it = 0; it < arr_out.ndim(); ++it)
            {
                auto step = slices[it][2];
                offset_out += arr_out.stride(it) * sidx[it] * step;
            }

            // NOLINTNEXTLINE(bugprone-signed-char-misuse, cert-str34-c)
            arr_out.at(offset_out) = static_cast<out_type>(*ptr_in);
            // recursion here
            copy_idx(arr_out, slices, arr_in, left_shape, sidx, dim - 1);
        }
    }

    static void broadcast(SimpleArray<T> & arr_out, std::vector<slice_type> const & slices, pybind11::array const & arr_in)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto * arr_new = reinterpret_cast<pybind11::array_t<D> const *>(&arr_in);

        shape_type left_shape(arr_out.ndim());
        for (size_t i = 0; i < arr_out.ndim(); i++)
        {
            slice_type const & slice = slices[i];
            if ((slice[1] - slice[0]) % slice[2] == 0)
            {
                left_shape[i] = (slice[1] - slice[0]) / slice[2];
            }
            else
            {
                left_shape[i] = (slice[1] - slice[0]) / slice[2] + 1;
            }
        }

        shape_type sidx_init(arr_out.ndim());

        for (size_t i = 0; i < arr_out.ndim(); ++i)
        {
            sidx_init[i] = 0;
        }

        copy_idx(arr_out, slices, arr_new, left_shape, sidx_init, static_cast<int>(arr_out.ndim()) - 1);
    }
}; /* end struct TypeBroadcastImpl */

template <typename T>
struct TypeBroadcast
{
    using slice_type = small_vector<int>;
    using shape_type = typename SimpleArray<T>::shape_type;

    static void check_shape(SimpleArray<T> const & arr_out, std::vector<slice_type> const & slices, pybind11::array const & arr_in)
    {
        shape_type right_shape(arr_in.ndim());
        for (pybind11::ssize_t i = 0; i < arr_in.ndim(); i++)
        {
            right_shape[i] = arr_in.shape(i);
        }

        shape_type left_shape(arr_out.ndim());
        // TODO: range check
        for (size_t i = 0; i < arr_out.ndim(); i++)
        {
            const slice_type & slice = slices[i];
            if ((slice[1] - slice[0]) % slice[2] == 0)
            {
                left_shape[i] = (slice[1] - slice[0]) / slice[2];
            }
            else
            {
                left_shape[i] = (slice[1] - slice[0]) / slice[2] + 1;
            }
        }

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
        else if (dtype_is_type<uint32_t>(arr_in))
        {
            TypeBroadcastImpl<T, uint32_t>::broadcast(arr_out, slices, arr_in);
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
