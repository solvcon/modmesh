/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/python/common.hpp> // Must be the first include.
#include <pybind11/stl.h> // Required for automatic conversion.

#include <modmesh/toggle/toggle.hpp>

#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace modmesh
{

namespace python
{

void import_numpy()
{
    auto local_import_numpy = []()
    {
        import_array2("cannot import numpy", false); // or numpy c api segfault.
        return true;
    };
    if (!local_import_numpy())
    {
        throw pybind11::error_already_set();
    }
}

Interpreter & Interpreter::instance()
{
    static Interpreter o;
    return o;
}

Interpreter & Interpreter::initialize()
{
    if (nullptr == m_interpreter)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        m_interpreter = new pybind11::scoped_interpreter(true, 0, nullptr, true);
    }
    return *this;
}

Interpreter & Interpreter::finalize()
{
    if (nullptr != m_interpreter)
    {
        // Py_Main and Py_BytesMain may finalize the interpreter before this is reached.
        if (0 != Py_IsInitialized())
        {
            delete m_interpreter;
        }
        m_interpreter = nullptr;
    }
    return *this;
}

Interpreter & Interpreter::setup_modmesh_path()
{
    // The hard-coded Python in C++ is difficult to debug. Any better way?
    std::string const cmd = R""""(def _set_modmesh_path():
    import os
    import sys
    filename = os.path.join('modmesh', '__init__.py')
    path = os.getcwd()
    while True:
        if os.path.exists(os.path.join(path, filename)):
            break
        if path == os.path.dirname(path):
            path = None
            break
        else:
            path = os.path.dirname(path)
    if path:
        sys.path.insert(0, path)
_set_modmesh_path())"""";
    pybind11::exec(cmd);
    return *this;
}

Interpreter & Interpreter::setup_process()
{
    // NOLINTNEXTLINE(misc-const-correctness)
    pybind11::object mod_sys = pybind11::module_::import("modmesh.system");
    CommandLineInfo const & cmdinfo = ProcessInfo::instance().command_line();
    mod_sys.attr("setup_process")(cmdinfo.python_argv());
    return *this;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int Interpreter::enter_main()
{
    // NOLINTNEXTLINE(misc-const-correctness)
    pybind11::object mod_sys = pybind11::module_::import("modmesh.system");
    CommandLineInfo const & cmdinfo = ProcessInfo::instance().command_line();
    return pybind11::cast<int>(mod_sys.attr("enter_main")(cmdinfo.python_argv()));
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): implicitly requires m_interpreter
void Interpreter::preload_module(std::string const & name)
{
    namespace py = pybind11;

    std::cerr << "Loading " << name << " ... ";
    bool load_failure = false;
    try
    {
        py::module_::import(name.c_str());
    }
    catch (const py::error_already_set & e)
    {
        if (std::string::npos == std::string(e.what()).find("ModuleNotFoundError"))
        {
            throw;
        }
        else // NOLINT(llvm-else-after-return,readability-else-after-return)
        {
            std::cerr << "fails" << std::endl;
            load_failure = true;
        }
    }
    if (!load_failure)
    {
        std::cerr << "succeeds" << std::endl;
        // Load into the namespace.
        std::ostringstream ms;
        ms << "import " << name;
        py::exec(ms.str());
    }
}

void Interpreter::preload_modules(std::vector<std::string> const & names)
{
    for (auto const & name : names)
    {
        preload_module(name);
    }
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Interpreter::exec_code(std::string const & code, std::string const & redirect_stdout_file_path, std::string const & redirect_stderr_file_path)
{
    // NOLINTNEXTLINE(misc-const-correctness)
    pybind11::object mod_sys = pybind11::module_::import("modmesh.system");
    mod_sys.attr("exec_code")(code, redirect_stdout_file_path, redirect_stderr_file_path);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
