/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray_complex(pybind11::module & mod)
{
    WrapSimpleArray<Complex<float>>::commit(mod, "SimpleArrayComplex64", "SimpleArrayComplex64");
    WrapSimpleArray<Complex<double>>::commit(mod, "SimpleArrayComplex128", "SimpleArrayComplex128");

    WrapSimpleCollector<Complex<float>>::commit(mod, "SimpleCollectorComplex64", "SimpleCollectorComplex64");
    WrapSimpleCollector<Complex<double>>::commit(mod, "SimpleCollectorComplex128", "SimpleCollectorComplex128");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
