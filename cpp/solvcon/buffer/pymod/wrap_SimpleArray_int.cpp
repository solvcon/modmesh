/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray_int(pybind11::module & mod)
{
    WrapSimpleArray<int8_t>::commit(mod, "SimpleArrayInt8", "SimpleArrayInt8");
    WrapSimpleArray<int16_t>::commit(mod, "SimpleArrayInt16", "SimpleArrayInt16");
    WrapSimpleArray<int32_t>::commit(mod, "SimpleArrayInt32", "SimpleArrayInt32");
    WrapSimpleArray<int64_t>::commit(mod, "SimpleArrayInt64", "SimpleArrayInt64");

    WrapSimpleCollector<int8_t>::commit(mod, "SimpleCollectorInt8", "SimpleCollectorInt8");
    WrapSimpleCollector<int16_t>::commit(mod, "SimpleCollectorInt16", "SimpleCollectorInt16");
    WrapSimpleCollector<int32_t>::commit(mod, "SimpleCollectorInt32", "SimpleCollectorInt32");
    WrapSimpleCollector<int64_t>::commit(mod, "SimpleCollectorInt64", "SimpleCollectorInt64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
