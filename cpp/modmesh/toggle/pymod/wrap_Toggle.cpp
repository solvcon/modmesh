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

    static std::string report(wrapped_type const & self);

    static pybind11::object getattr(wrapped_type const & self, std::string const & key)
    {
        namespace py = pybind11;

        DynamicToggleIndex const index = self.get_dynamic_index(key);
        switch (index.type)
        {
        case DynamicToggleIndex::TYPE_NONE:
            throw py::attribute_error(Formatter() << "Cannt get non-existing key \"" << key << "\"");
            break;
        case DynamicToggleIndex::TYPE_BOOL:
            return py::cast(self.get_bool(key));
            break;
        case DynamicToggleIndex::TYPE_INT8:
            return py::cast(self.get_int8(key));
            break;
        case DynamicToggleIndex::TYPE_INT16:
            return py::cast(self.get_int16(key));
            break;
        case DynamicToggleIndex::TYPE_INT32:
            return py::cast(self.get_int32(key));
            break;
        case DynamicToggleIndex::TYPE_INT64:
            return py::cast(self.get_int64(key));
            break;
        case DynamicToggleIndex::TYPE_REAL:
            return py::cast(self.get_real(key));
            break;
        case DynamicToggleIndex::TYPE_STRING:
            return py::cast(self.get_string(key));
            break;
        default:
            return py::none();
            break;
        }
    }

    static void setattr(wrapped_type & self, std::string const & key, pybind11::object & value)
    {
        namespace py = pybind11;

        DynamicToggleIndex const index = self.get_dynamic_index(key);
        switch (index.type)
        {
        case DynamicToggleIndex::TYPE_NONE:
            /* It is intentional to throw an exception when the key does not
             * exist.  Key-value pairs in the toggle object are supposed to be
             * added using the set_TYPE() functions, not the Pythonic
             * __setattr__().
             *
             * Do not try to "fix" the exception using RTTI. */
            throw pybind11::attribute_error(Formatter() << "Cannot set non-existing key \"" << key << "\"; use set_TYPE() instead");
            break;
        case DynamicToggleIndex::TYPE_BOOL:
            self.set_bool(key, py::cast<bool>(value));
            break;
        case DynamicToggleIndex::TYPE_INT8:
            self.set_int8(key, py::cast<int8_t>(value));
            break;
        case DynamicToggleIndex::TYPE_INT16:
            self.set_int16(key, py::cast<int16_t>(value));
            break;
        case DynamicToggleIndex::TYPE_INT32:
            self.set_int32(key, py::cast<int32_t>(value));
            break;
        case DynamicToggleIndex::TYPE_INT64:
            self.set_int64(key, py::cast<int64_t>(value));
            break;
        case DynamicToggleIndex::TYPE_REAL:
            self.set_real(key, py::cast<double>(value));
            break;
        case DynamicToggleIndex::TYPE_STRING:
            self.set_string(key, py::cast<std::string>(value));
            break;
        default:
            break;
        }
    }

}; /* end class WrapToggle */

WrapToggle::WrapToggle(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def("report", &report)
        //
        ;

    // Dynamic properties.  Number of the properties can be freely changed
    // during runtime.
    (*this)
        .def("__getattr__", getattr)
        .def("__setattr__", setattr)
        .def("dynamic_keys", &wrapped_type::dynamic_keys)
        .def("dynamic_clear", &wrapped_type::dynamic_clear)
        .def("get_bool", &wrapped_type::get_bool, py::arg("key"))
        .def("set_bool", &wrapped_type::set_bool, py::arg("key"), py::arg("value"))
        .def("get_int8", &wrapped_type::get_int8, py::arg("key"))
        .def("set_int8", &wrapped_type::set_int8, py::arg("key"), py::arg("value"))
        .def("get_int16", &wrapped_type::get_int16, py::arg("key"))
        .def("set_int16", &wrapped_type::set_int16, py::arg("key"), py::arg("value"))
        .def("get_int32", &wrapped_type::get_int32, py::arg("key"))
        .def("set_int32", &wrapped_type::set_int32, py::arg("key"), py::arg("value"))
        .def("get_int64", &wrapped_type::get_int64, py::arg("key"))
        .def("set_int64", &wrapped_type::set_int64, py::arg("key"), py::arg("value"))
        .def("get_real", &wrapped_type::get_real, py::arg("key"))
        .def("set_real", &wrapped_type::set_real, py::arg("key"), py::arg("value"))
        .def("get_string", &wrapped_type::get_string, py::arg("key"))
        .def("set_string", &wrapped_type::set_string, py::arg("key"), py::arg("value"))
        //
        ;

    // Static properties.
    (*this)
        .def_property_readonly_static(
            "instance",
            [](py::object const &) -> auto &
            { return wrapped_type::instance(); })
        //
        ;

    // Instance properties.
    (*this)
        .def_property_readonly(
            "use_pyside",
            [](wrapped_type const & self)
            { return self.fixed().get_use_pyside(); })
        .def_property(
            "show_axis",
            [](wrapped_type const & self)
            { return self.fixed().get_show_axis(); },
            [](wrapped_type & self, bool v)
            { self.fixed().set_show_axis(v); })
        //
        ;
}

std::string WrapToggle::report(WrapToggle::wrapped_type const & self)
{
    Formatter ret;
    ret << "Toggle: "
        << "USE_PYSIDE=" << self.fixed().get_use_pyside();
    return ret >> Formatter::to_str;
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
            [](py::object const &) -> auto &
            {
                return wrapped_type::instance();
            })
        .def(
            "set_environment_variables",
            [](wrapped_type & self) -> auto &
            {
                return self.set_environment_variables();
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "command_line",
            [](wrapped_type & self) -> auto &
            {
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
