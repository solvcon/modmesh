/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray_bool(pybind11::module & mod)
{
    WrapSimpleArray<bool>::commit(mod, "SimpleArrayBool", "SimpleArrayBool");

    WrapSimpleCollector<bool>::commit(mod, "SimpleCollectorBool", "SimpleCollectorBool");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
