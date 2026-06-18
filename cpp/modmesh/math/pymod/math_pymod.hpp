/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>

#include <modmesh/python/common.hpp>

namespace modmesh
{

namespace python
{

void initialize_math(pybind11::module & mod);
void wrap_Complex(pybind11::module & mod);

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
