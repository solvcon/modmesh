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

#include <pybind11/pybind11.h> // Must be the first include.

#include <modmesh/buffer/SimpleArray.hpp>
#include <modmesh/buffer/pymod/TypeBroadcast.hpp>
#include <modmesh/math/math.hpp>

// We faced an issue where the template specialization for the caster of
// SimpleArray<T> doesn't function correctly on both macOS and Windows.
// While the root cause of the problem remains unclear, a workaround is
// available by including the caster header in this file, impacting
// wrap_SimpleArray.cpp.
// See more details in the issue: https://github.com/solvcon/modmesh/issues/283
#include <modmesh/buffer/pymod/SimpleArrayCaster.hpp>

namespace pybind11
{

namespace detail
{

template <>
struct npy_format_descriptor<modmesh::Complex<double>>
{
    static constexpr auto name = const_name("complex128");
    static constexpr int value = npy_api::NPY_CDOUBLE_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex128");
    }

    // The format string is used by numpy to correctly interpret the memory layout
    // of Complex<T> when converting between c++ and python.
    static std::string format()
    {
        return "=Zd";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(typename std::remove_cv<modmesh::Complex<double>>::type),
                                  sizeof(modmesh::Complex<double>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        static PyObject * ptr = get_numpy_internals().get_type_info<modmesh::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& value)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                value = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval;
                return true;
            }
        }
        return false;
    }
};

template <>
struct npy_format_descriptor<modmesh::Complex<float>>
{
    static constexpr auto name = const_name("complex64");
    static constexpr int value = npy_api::NPY_CFLOAT_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex64");
    }

    static std::string format()
    {
        return "=Zf";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(typename std::remove_cv<modmesh::Complex<float>>::type),
                                  sizeof(modmesh::Complex<float>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        static PyObject * ptr = get_numpy_internals().get_type_info<modmesh::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& value)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                value = (reinterpret_cast<PyVoidScalarObject_Proxy *>(obj))->obval;
                return true;
            }
        }
        return false;
    }
};

} /* end namespace detail */

} /* end namespace pybind11 */

namespace modmesh
{

namespace python
{

inline modmesh::detail::shape_type make_shape(pybind11::object const & shape_in)
{
    modmesh::detail::shape_type shape;
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

/// Helper class for array property in Python.
template <typename T>
class ArrayPropertyHelper
{
public:
    using shape_type = modmesh::detail::shape_type;
    using slice_type = modmesh::detail::slice_type;

    static void broadcast_array_using_ellipsis(SimpleArray<T> & arr_out, pybind11::array const & arr_in)
    {
        auto slices = make_default_slices(arr_out);

        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);

        const size_t nghost = arr_out.nghost();
        if (0 != nghost)
        {
            arr_out.set_nghost(0);
        }

        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);

        if (0 != nghost)
        {
            arr_out.set_nghost(nghost);
        }
    }

    static void setitem_parser(SimpleArray<T> & arr_out, pybind11::args const & args)
    {
        namespace py = pybind11;

        if (args.size() == 2)
        {
            const py::object & py_key = args[0];
            const py::object & py_value = args[1];

            const bool is_number = py::isinstance<py::bool_>(py_value) || py::isinstance<py::int_>(py_value) || py::isinstance<py::float_>(py_value) || is_complex_v<T>;

            // sarr[K] = V
            if (py::isinstance<py::int_>(py_key) && is_number)
            {
                const auto key = py_key.cast<ssize_t>();

                arr_out.at(key) = py_value.cast<T>();
                return;
            }
            // sarr[K1, K2, K3] = V
            if (py::isinstance<py::tuple>(py_key) && is_number)
            {
                const auto key = py_key.cast<std::vector<ssize_t>>();

                arr_out.at(key) = py_value.cast<T>();
                return;
            }

            const bool is_sequence = py::isinstance<py::list>(py_value) || py::isinstance<py::array>(py_value) || py::isinstance<py::tuple>(py_value);

            // multi-dimension with slice and ellipsis
            // sarr[slice, slice, ellipsis] = ndarr
            if (py::isinstance<py::tuple>(py_key) && is_sequence)
            {
                const py::tuple tuple_in = py_key;
                const py::array arr_in = py_value;

                auto slices = make_default_slices(arr_out);
                process_slices(tuple_in, slices, arr_out.ndim());

                broadcast_array_using_slice(arr_out, slices, arr_in);
                return;
            }
            // one-dimension with slice
            // sarr[slice] = ndarr
            if (py::isinstance<py::slice>(py_key) && is_sequence)
            {
                const auto slice_in = py_key.cast<py::slice>();
                const auto arr_in = py_value.cast<py::array>();

                auto slices = make_default_slices(arr_out);
                copy_slice(slices[0], slice_in);

                broadcast_array_using_slice(arr_out, slices, arr_in);
                return;
            }
            // sarr[ellipsis] = ndarr
            if (py::isinstance<py::ellipsis>(py_key) && is_sequence)
            {
                const auto arr_in = py_value.cast<py::array>();

                broadcast_array_using_ellipsis(arr_out, arr_in);
                return;
            }
        }
        throw std::runtime_error("unsupported operation.");
    }

    static pybind11::buffer_info get_buffer_info(SimpleArray<T> & array)
    {
        std::vector<size_t> stride;
        for (size_t const i : array.stride())
        {
            stride.push_back(i * sizeof(T));
        }

        // Special handling for Complex types
        std::string format;
        if constexpr (is_complex_v<T>)
        {
            if constexpr (std::is_same_v<T, Complex<double>>)
            {
                format = pybind11::format_descriptor<Complex<double>>::format();
            }
            else
            {
                format = pybind11::format_descriptor<Complex<float>>::format();
            }
        }
        else
        {
            format = pybind11::format_descriptor<T>::format();
        }

        return pybind11::buffer_info(
            array.data(), /* Pointer to buffer */
            sizeof(T), /* Size of one scalar */
            format, /* Python struct-style format descriptor */
            array.ndim(), /* Number of dimensions */
            std::vector<size_t>(array.shape().begin(), array.shape().end()), /* Buffer dimensions */
            stride /* Strides (in bytes) for each index */
        );
    }

private:

    static std::vector<slice_type> make_default_slices(SimpleArray<T> const & arr)
    {
        std::vector<slice_type> slices;
        slices.reserve(arr.ndim());
        for (size_t i = 0; i < arr.ndim(); ++i)
        {
            slice_type default_slice(3);
            default_slice[0] = 0; // start
            default_slice[1] = static_cast<int>(arr.shape(i)); // stop
            default_slice[2] = 1; // step
            slices.push_back(std::move(default_slice));
        }
        return slices;
    }

    static void copy_slice(slice_type & slice_out, pybind11::slice const & slice_in)
    {
        auto start = std::string(pybind11::str(slice_in.attr("start")));
        auto stop = std::string(pybind11::str(slice_in.attr("stop")));
        auto step = std::string(pybind11::str(slice_in.attr("step")));

        slice_out[0] = start == "None" ? slice_out[0] : std::stoi(start);
        slice_out[1] = stop == "None" ? slice_out[1] : std::stoi(stop);
        slice_out[2] = step == "None" ? slice_out[2] : std::stoi(step);
    }

    static void slice_syntax_check(pybind11::tuple const & tuple, size_t ndim)
    {
        namespace py = pybind11;

        size_t ellipsis_cnt = 0;
        size_t slice_cnt = 0;

        for (auto it = tuple.begin(); it != tuple.end(); it++)
        {
            if (py::isinstance<py::ellipsis>(*it))
            {
                ellipsis_cnt += 1;
            }
            else if (py::isinstance<py::slice>(*it))
            {
                slice_cnt += 1;
            }
            else
            {
                throw std::runtime_error("unsupported operation.");
            }
        }

        if (ellipsis_cnt + slice_cnt > ndim)
        {
            throw std::runtime_error("syntax error. dimensions mismatches");
        }

        if (ellipsis_cnt > 1)
        {
            throw std::runtime_error("syntax error. no more than one ellipsis.");
        }
    }

    static void process_slices(pybind11::tuple const & tuple,
                               std::vector<slice_type> & slices,
                               size_t ndim)
    {
        namespace py = pybind11;

        // copy slices from the front until an ellipsis
        bool ellipsis_flag = false;
        for (auto it = tuple.begin(); it != tuple.end(); it++)
        {
            if (py::isinstance<py::ellipsis>(*it))
            {
                // stop here and iterator the tuple from back later
                ellipsis_flag = true;
                break;
            }

            auto & slice_out = slices[it - tuple.begin()];
            const auto slice_in = (*it).cast<py::slice>();

            copy_slice(slice_out, slice_in);
        }

        // copy slices from the back until an ellipsis
        if (ellipsis_flag)
        {
            for (size_t size = 0; size < tuple.size(); size++)
            {
                auto it = tuple.end() - size - 1;

                if (py::isinstance<py::ellipsis>(*it))
                {
                    break;
                }
                auto & slice_out = slices[ndim - size - 1];
                const auto slice_in = (*it).cast<py::slice>();

                copy_slice(slice_out, slice_in);
            }
        }
    }

    static void broadcast_array_using_slice(SimpleArray<T> & arr_out,
                                            std::vector<slice_type> const & slices,
                                            pybind11::array const & arr_in)
    {
        TypeBroadcast<T>::check_shape(arr_out, slices, arr_in);

        const size_t nghost = arr_out.nghost();
        if (0 != nghost)
        {
            arr_out.set_nghost(0);
        }

        TypeBroadcast<T>::broadcast(arr_out, slices, arr_in);

        if (0 != nghost)
        {
            arr_out.set_nghost(nghost);
        }
    }
};

} /* end namespace python */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
