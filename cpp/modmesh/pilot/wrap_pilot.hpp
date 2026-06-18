#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/python/python.hpp> // Must be the first include.

namespace modmesh
{

namespace python
{

void initialize_pilot(pybind11::module & mod);
void wrap_pilot(pybind11::module & mod);

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
