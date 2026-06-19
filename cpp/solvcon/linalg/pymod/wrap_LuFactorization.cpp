/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>

#include <solvcon/linalg/pymod/linalg_pymod.hpp>

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapLuFactorization
    : public WrapBase<WrapLuFactorization<T>, LuFactorization<T>>
{

    using root_base_type = WrapBase<WrapLuFactorization<T>, LuFactorization<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend root_base_type;

    WrapLuFactorization(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapLuFactorization */

template <typename T>
WrapLuFactorization<T>::WrapLuFactorization(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](array_type const & a)
                {
                    return std::make_unique<wrapped_type>(a);
                }),
            py::arg("a"));

    (*this)
        .def_property_readonly(
            "lu",
            &wrapped_type::lu,
            py::return_value_policy::reference_internal)
        // piv is a SimpleArray<int64_t>; the SimpleArray caster maps it to a
        // SimpleArrayInt64 that shares the cached buffer, so tie its lifetime
        // to the owner like lu above.
        .def_property_readonly(
            "piv",
            &wrapped_type::piv,
            py::return_value_policy::reference_internal)
        .def_property_readonly("n", &wrapped_type::n)
        .def(
            "solve",
            &wrapped_type::solve,
            py::arg("b"),
            "Solve A x = b using the cached LU factors.")
        .def(
            "inv",
            &wrapped_type::inv,
            "Compute A^(-1) using the cached LU factors.")
        .def(
            "det",
            [](wrapped_type const & self)
            {
                // Complex<T> has no pybind11 caster; convert to std::complex
                // (which pybind11 maps to a Python complex) for complex T.
                if constexpr (is_complex_v<T>)
                {
                    return self.det().to_std_complex();
                }
                else
                {
                    return self.det();
                }
            },
            "Compute det(A) using the cached LU factors.");
}

void wrap_LuFactorization(pybind11::module & mod)
{
    WrapLuFactorization<float>::commit(
        mod, "LuFactorizationFloat32", "LU factorization (float32)");
    WrapLuFactorization<double>::commit(
        mod, "LuFactorizationFloat64", "LU factorization (float64)");
    WrapLuFactorization<Complex<float>>::commit(
        mod, "LuFactorizationComplex64", "LU factorization (complex64)");
    WrapLuFactorization<Complex<double>>::commit(
        mod, "LuFactorizationComplex128", "LU factorization (complex128)");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
