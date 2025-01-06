/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.
 */

#include <modmesh/python/python.hpp> // Must be the first include.

#include <modmesh/buffer/pymod/buffer_pymod.hpp>
#include <modmesh/inout/pymod/inout_pymod.hpp>
#include <modmesh/mesh/pymod/mesh_pymod.hpp>
#include <modmesh/onedim/pymod/onedim_pymod.hpp>
#include <modmesh/multidim/pymod/multidim_pymod.hpp>
#include <modmesh/python/module.hpp>
#include <modmesh/spacetime/pymod/spacetime_pymod.hpp>
#include <modmesh/toggle/pymod/toggle_pymod.hpp>
#include <modmesh/universe/pymod/universe_pymod.hpp>
#include <modmesh/math/pymod/math_pymod.hpp>

#ifdef USE_PYTEST_HELPER_BINDING
#include <modmesh/testhelper/pymod/testbuffer_pymod.hpp>
#endif // USE_PYTEST_HELPER_BINDING

#ifdef QT_CORE_LIB
#include <modmesh/pilot/wrap_pilot.hpp>
#endif // QT_CORE_LIB

namespace modmesh
{

namespace python
{

void initialize(pybind11::module_ mod)
{
    initialize_toggle(mod);
    initialize_buffer(mod);
    initialize_universe(mod);
    initialize_mesh(mod);
    initialize_multidim(mod);
    initialize_inout(mod);
    initialize_math(mod);
    pybind11::module_ spacetime_mod = mod.def_submodule("spacetime", "spacetime");
    initialize_spacetime(spacetime_mod);
    pybind11::module_ onedim_mod = mod.def_submodule("onedim", "onedim");
    initialize_onedim(onedim_mod);

    pybind11::module_ testhelper_mod = mod.def_submodule("testhelper", "testhelper");
#ifdef USE_PYTEST_HELPER_BINDING
    initialize_testbuffer(testhelper_mod);
    testhelper_mod.attr("PYTEST_HELPER_BINDING_BUILT") = true;
#else
    testhelper_mod.attr("PYTEST_HELPER_BINDING_BUILT") = false;
#endif

#ifdef QT_CORE_LIB
    mod.attr("HAS_PILOT") = true;
    pybind11::module_ view_mod = mod.def_submodule("pilot", "pilot");
    initialize_pilot(view_mod);
#else // QT_CORE_LIB
    mod.attr("HAS_PILOT") = false;
#endif // QT_CORE_LIB
}

int program_entrance(int argc, char ** argv)
{
    ProcessInfo::instance().populate_command_line(argc, argv);
    ProcessInfo::instance().set_environment_variables();
    auto & clinfo = ProcessInfo::instance().command_line();

    // Initialize the Python interpreter.
    Interpreter::instance()
        .initialize()
        .setup_modmesh_path()
        .setup_process();

    int ret = 0;

    if (clinfo.python_main())
    {
        ret = Py_BytesMain(clinfo.python_main_argc(), clinfo.python_main_argv_ptr());
    }
    else
    {
        ret = Interpreter::instance().enter_main();
    }

    Interpreter::instance().finalize();
    return ret;
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
