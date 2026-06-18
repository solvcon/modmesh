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

void initialize(pybind11::module_ mod);
int program_entrance(int argc, char ** argv);

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
