/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/linalg/pymod/linalg_pymod.hpp>

namespace solvcon
{

namespace python
{

struct linalg_pymod_tag;

template <>
OneTimeInitializer<linalg_pymod_tag> & OneTimeInitializer<linalg_pymod_tag>::me()
{
    static OneTimeInitializer<linalg_pymod_tag> instance;
    return instance;
}

void initialize_linalg(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_factorization(mod);
        wrap_states_info(mod);
        wrap_kalman_filter(mod);
        wrap_EigenSystem(mod);
        wrap_LuFactorization(mod);
    };

    OneTimeInitializer<linalg_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
