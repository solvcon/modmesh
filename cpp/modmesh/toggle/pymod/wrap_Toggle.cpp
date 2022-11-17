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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/toggle/pymod/toggle_pymod.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

#ifdef MODMESH_METAL
#include <modmesh/device/metal/metal.hpp>
#endif // MODMESH_METAL

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapToggle
    : public WrapBase<WrapToggle, Toggle>
{

public:

    using base_type = WrapBase<WrapToggle, Toggle>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapToggle(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapToggle */

WrapToggle::WrapToggle(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;
    (*this)
        .def_property_readonly_static(
            "instance",
            [](py::object const &) -> auto & {
                return wrapped_type::instance();
            })
        .def_property("show_axis", &wrapped_type::get_show_axis, &wrapped_type::set_show_axis);
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapCommandLineInfo
    : public WrapBase<WrapCommandLineInfo, CommandLineInfo>
{

public:

    using base_type = WrapBase<WrapCommandLineInfo, CommandLineInfo>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapCommandLineInfo(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapCommandLineInfo */

WrapCommandLineInfo::WrapCommandLineInfo(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    (*this)
        .def("freeze", &wrapped_type::freeze)
        .def_property_readonly("frozen", &wrapped_type::frozen)
        .def_property_readonly("populated", &wrapped_type::populated)
        .def_property_readonly("populated_argv", &wrapped_type::populated_argv)
        .def_property_readonly("executable_basename", &wrapped_type::executable_basename)
        .def_property("python_argv", &wrapped_type::python_argv, &wrapped_type::set_python_argv);
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapProcessInfo
    : public WrapBase<WrapProcessInfo, ProcessInfo>
{

public:

    using base_type = WrapBase<WrapProcessInfo, ProcessInfo>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapProcessInfo(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapProcessInfo */

WrapProcessInfo::WrapProcessInfo(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;
    (*this)
        .def_property_readonly_static(
            "instance",
            [](py::object const &) -> auto & {
                return wrapped_type::instance();
            })
        .def_property_readonly(
            "command_line",
            [](wrapped_type & self) -> auto & {
                return self.command_line();
            },
            py::return_value_policy::reference_internal);
}

void wrap_Toggle(pybind11::module & mod)
{
    WrapToggle::commit(mod, "Toggle", "Toggle");
    WrapCommandLineInfo::commit(mod, "CommandLineInfo", "CommandLineInfo");
    WrapProcessInfo::commit(mod, "ProcessInfo", "ProcessInfo");

#ifdef MODMESH_METAL
    mod.attr("METAL_BUILT") = true;
#else // MODMESH_METAL
    mod.attr("METAL_BUILT") = false;
#endif // MODMESH_METAL
    mod.def(
        "metal_running",
        []()
        {
#ifdef MODMESH_METAL
            return ::modmesh::device::MetalManager::instance().started();
#else // MODMESH_METAL
            return false;
#endif // MODMESH_METAL
        });
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
