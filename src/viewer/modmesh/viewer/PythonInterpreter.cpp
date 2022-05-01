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

#include <modmesh/viewer/base.hpp> // Must be the first include.

#include <modmesh/viewer/PythonInterpreter.hpp>

#include <modmesh/viewer/RMainWindow.hpp>
#include <modmesh/viewer/RStaticMesh.hpp>

namespace modmesh
{

PythonInterpreter & PythonInterpreter::instance()
{
    static PythonInterpreter o;
    return o;
}

PythonInterpreter::PythonInterpreter()
    : m_interpreter(new pybind11::scoped_interpreter)
{
    load_modules();
}

PythonInterpreter::~PythonInterpreter()
{
    if (m_interpreter)
    {
        delete m_interpreter;
    }
}

void PythonInterpreter::load_modules()
{
    namespace py = pybind11;

    {
        // TODO: The hard-coded Python in C++ is difficult to debug.  This
        // should be moved somewhere else in the future.
        std::string cmd = R""""(def _set_modmesh_path():
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
        py::exec(cmd);
        // clang-format off
    }
    // clang-format on

    // Load the Python extension modules.
    std::vector<std::string> modules{"_modmesh_view", "modmesh"};

    for (auto const & mod : modules)
    {
        std::cerr << "Loading " << mod << " ... ";
        bool load_failure = false;
        try
        {
            py::module_::import(mod.c_str());
        }
        catch (const py::error_already_set & e)
        {
            if (std::string::npos == std::string(e.what()).find("ModuleNotFoundError"))
            {
                throw;
            }
            else
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
            ms << "import " << mod;
            py::exec(ms.str());
        }
    }

    py::exec("modmesh.view = _modmesh_view");
}

} /* end namespace modmesh */

PYBIND11_EMBEDDED_MODULE(_modmesh_view, mod)
{
    using namespace modmesh;
    namespace py = pybind11;

    mod
        .def(
            "show",
            [](std::shared_ptr<StaticMesh2d> const & mesh)
            {
                RApplication * app = RApplication::instance();
                RScene * scene = app->main()->viewer()->scene();
                for (RStaticMesh<2> * child : scene->findChildren<RStaticMesh<2> *>())
                {
                    delete child;
                }
                new RStaticMesh<2>(mesh, app->main()->viewer()->scene());
            });
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
