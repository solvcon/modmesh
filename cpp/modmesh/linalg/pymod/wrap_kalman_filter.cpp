/*
 * Copyright (c) 2025, Chun-Shih Chang <austin20463@gmail.com>
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

#include <modmesh/modmesh.hpp>
#include <modmesh/linalg/pymod/linalg_pymod.hpp>
#include <pybind11/operators.h>

namespace modmesh
{

namespace python
{

template <typename T>
class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapKalmanFilter
    : public WrapBase<WrapKalmanFilter<T>, KalmanFilter<T>>
{
    using base_type = WrapBase<WrapKalmanFilter<T>, KalmanFilter<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using array_type = SimpleArray<T>;
    using real_type = typename wrapped_type::real_type;

    friend base_type;

    WrapKalmanFilter(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](array_type const & x, array_type const & f, py::object const & b, array_type const & h, real_type process_noise, real_type measurement_noise, real_type jitter)
                    {
                        array_type b_array;
                        if (b.is_none())
                        {
                            b_array = array_type(small_vector<size_t>{x.shape(0), 0});
                        }
                        else
                        {
                            b_array = b.cast<array_type>();
                        }
                        return wrapped_type(x, f, b_array, h, process_noise, measurement_noise, jitter);
                    }),
                py::arg("x"),
                py::arg("f"),
                py::arg("b") = py::none(),
                py::arg("h"),
                py::arg("process_noise"),
                py::arg("measurement_noise"),
                py::arg("jitter") = static_cast<real_type>(1e-9))
            .def_property_readonly(
                "state",
                &wrapped_type::state)
            .def_property_readonly(
                "covariance",
                &wrapped_type::covariance)
            .def(
                "predict",
                [](wrapped_type & self)
                { self.predict(); })
            .def(
                "predict",
                [](wrapped_type & self, array_type const & u)
                { self.predict(u); },
                py::arg("u"))
            .def(
                "update",
                &wrapped_type::update,
                py::arg("z"));
    }

}; /* end class WrapKalmanFilter */

void wrap_kalman_filter(pybind11::module & mod)
{
    WrapKalmanFilter<float>::commit(mod, "KalmanFilterFp32", "Kalman Filter (float)");
    WrapKalmanFilter<double>::commit(mod, "KalmanFilterFp64", "Kalman Filter (double)");
    WrapKalmanFilter<Complex<float>>::commit(mod, "KalmanFilterComplex64", "Kalman Filter (complex float)");
    WrapKalmanFilter<Complex<double>>::commit(mod, "KalmanFilterComplex128", "Kalman Filter (complex double)");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
