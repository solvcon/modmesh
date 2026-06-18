/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/linalg/pymod/linalg_pymod.hpp>

namespace modmesh
{

namespace python
{

template <typename T>
void def_llt_factorization(pybind11::module & mod)
{
    mod.def(
        "llt_factorization", [](SimpleArray<T> const & a)
        { return llt_factorization(a); },
        pybind11::arg("a"));
}

template <typename T>
void def_llt_solve(pybind11::module & mod)
{
    mod.def(
        "llt_solve", [](SimpleArray<T> const & a, SimpleArray<T> const & b)
        { return llt_solve(a, b); },
        pybind11::arg("a"),
        pybind11::arg("b"));
}

template <typename T>
void def_lu_factorization(pybind11::module & mod)
{
    mod.def(
        "lu_factorization", [](SimpleArray<T> const & a)
        {
            auto [lu, piv] = lu_factorization(a);
            return pybind11::make_tuple(lu, piv); },
        pybind11::arg("a"));
}

// Python binding for lu_solve().  Solves Ax = b for general matrices.
template <typename T>
void def_lu_solve(pybind11::module & mod)
{
    mod.def(
        "lu_solve", [](SimpleArray<T> const & a, SimpleArray<T> const & b)
        { return lu_solve(a, b); },
        pybind11::arg("a"),
        pybind11::arg("b"));
}

// Python binding for lu_inv().  Computes A^(-1) for general matrices.
template <typename T>
void def_lu_inv(pybind11::module & mod)
{
    mod.def(
        "lu_inv", [](SimpleArray<T> const & a)
        { return lu_inv(a); },
        pybind11::arg("a"));
}

// Python binding for lu_det().  Computes det(A) for general matrices.
template <typename T>
void def_lu_det(pybind11::module & mod)
{
    mod.def(
        "lu_det", [](SimpleArray<T> const & a)
        {
            // Complex<T> has no pybind11 caster; convert to std::complex
            // (which pybind11 maps to a Python complex) for complex T.
            if constexpr (is_complex_v<T>)
            {
                return lu_det(a).to_std_complex();
            }
            else
            {
                return lu_det(a);
            } },
        pybind11::arg("a"));
}

// Attach .solve() and .inv() methods to an already-registered SimpleArray
// class.  The buffer module (where SimpleArray is defined) intentionally has
// no dependency on linalg/, so LU-based methods are injected here from the
// linalg module after the class has been registered.
template <typename T>
void add_simple_array_lu_methods(pybind11::module & mod, char const * pyname)
{
    namespace py = pybind11;
    py::object cls = mod.attr(pyname); // NOLINT(misc-const-correctness)
    cls.attr("solve") = py::cpp_function(
        [](SimpleArray<T> const & self, SimpleArray<T> const & b)
        { return lu_solve(self, b); },
        py::arg("b"),
        py::is_method(cls),
        "Solve linear system self @ x = b, returns x");
    cls.attr("inv") = py::cpp_function(
        [](SimpleArray<T> const & self)
        { return lu_inv(self); },
        py::is_method(cls),
        "Compute matrix inverse");
    cls.attr("det") = py::cpp_function(
        [](SimpleArray<T> const & self)
        {
            // Complex<T> has no pybind11 caster; convert to std::complex
            // (which pybind11 maps to a Python complex) for complex T.
            if constexpr (is_complex_v<T>)
            {
                return lu_det(self).to_std_complex();
            }
            else
            {
                return lu_det(self);
            }
        },
        py::is_method(cls),
        "Compute matrix determinant");
}

void wrap_factorization(pybind11::module & mod)
{
    def_llt_factorization<float>(mod);
    def_llt_factorization<double>(mod);
    def_llt_factorization<Complex<float>>(mod);
    def_llt_factorization<Complex<double>>(mod);

    def_llt_solve<float>(mod);
    def_llt_solve<double>(mod);
    def_llt_solve<Complex<float>>(mod);
    def_llt_solve<Complex<double>>(mod);

    def_lu_factorization<float>(mod);
    def_lu_factorization<double>(mod);
    def_lu_factorization<Complex<float>>(mod);
    def_lu_factorization<Complex<double>>(mod);

    def_lu_solve<float>(mod);
    def_lu_solve<double>(mod);
    def_lu_solve<Complex<float>>(mod);
    def_lu_solve<Complex<double>>(mod);

    def_lu_inv<float>(mod);
    def_lu_inv<double>(mod);
    def_lu_inv<Complex<float>>(mod);
    def_lu_inv<Complex<double>>(mod);

    def_lu_det<float>(mod);
    def_lu_det<double>(mod);
    def_lu_det<Complex<float>>(mod);
    def_lu_det<Complex<double>>(mod);

    // Integer arrays are deliberately excluded: LU decomposition involves
    // division, which would silently truncate under integer arithmetic.
    add_simple_array_lu_methods<float>(mod, "SimpleArrayFloat32");
    add_simple_array_lu_methods<double>(mod, "SimpleArrayFloat64");
    add_simple_array_lu_methods<Complex<float>>(mod, "SimpleArrayComplex64");
    add_simple_array_lu_methods<Complex<double>>(mod, "SimpleArrayComplex128");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
