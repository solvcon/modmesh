/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray_uint(pybind11::module & mod)
{
    WrapSimpleArray<uint8_t>::commit(mod, "SimpleArrayUint8", "SimpleArrayUint8");
    WrapSimpleArray<uint16_t>::commit(mod, "SimpleArrayUint16", "SimpleArrayUint16");
    WrapSimpleArray<uint32_t>::commit(mod, "SimpleArrayUint32", "SimpleArrayUint32");
    WrapSimpleArray<uint64_t>::commit(mod, "SimpleArrayUint64", "SimpleArrayUint64");

    WrapSimpleCollector<uint8_t>::commit(mod, "SimpleCollectorUint8", "SimpleCollectorUint8");
    WrapSimpleCollector<uint16_t>::commit(mod, "SimpleCollectorUint16", "SimpleCollectorUint16");
    WrapSimpleCollector<uint32_t>::commit(mod, "SimpleCollectorUint32", "SimpleCollectorUint32");
    WrapSimpleCollector<uint64_t>::commit(mod, "SimpleCollectorUint64", "SimpleCollectorUint64");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
