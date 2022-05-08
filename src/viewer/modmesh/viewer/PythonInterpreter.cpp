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
#include <modmesh/viewer/R3DWidget.hpp>

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

namespace python
{

// clang-format off
class
MODMESH_PYTHON_WRAPPER_VISIBILITY
WrapR3DWidget
    // clang-format on
    : public WrapBase<WrapR3DWidget, R3DWidget>
{

    friend root_base_type;

    WrapR3DWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

#define DECL_QVECTOR3D_PROPERTY(NAME, GETTER, SETTER)          \
    .def_property(                                             \
        #NAME,                                                 \
        [](wrapped_type & self)                                \
        {                                                      \
            QVector3D const v = self.camera()->GETTER();       \
            return py::make_tuple(v.x(), v.y(), v.z());        \
        },                                                     \
        [](wrapped_type & self, std::vector<double> const & v) \
        {                                                      \
            double const x = v.at(0);                          \
            double const y = v.at(1);                          \
            double const z = v.at(2);                          \
            self.camera()->SETTER(QVector3D(x, y, z));         \
        })

        (*this)
            DECL_QVECTOR3D_PROPERTY(position, position, setPosition)
                DECL_QVECTOR3D_PROPERTY(up_vector, upVector, setUpVector)
                    DECL_QVECTOR3D_PROPERTY(view_center, viewCenter, setViewCenter)
            //
            ;

#undef DECL_QVECTOR3D_PROPERTY
    }

}; /* end class WrapR3DWidget */

class
    MODMESH_PYTHON_WRAPPER_VISIBILITY
        WrapRApplication
    // clang-format on
    : public WrapBase<WrapRApplication, RApplication>
{

    friend root_base_type;

    WrapRApplication(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &)
                {
                    return RApplication::instance();
                })
            .def_property_readonly(
                "viewer",
                [](wrapped_type & self)
                {
                    return self.main()->viewer();
                })
            //
            ;
    }

}; /* end class WrapRApplication */

namespace detail
{

static void update_appmesh(std::shared_ptr<StaticMesh> const & mesh)
{
    RScene * scene = RApplication::instance()->main()->viewer()->scene();
    for (Qt3DCore::QNode * child : scene->childNodes())
    {
        if (typeid(*child) == typeid(RStaticMesh))
        {
            child->deleteLater();
        }
    }
    new RStaticMesh(mesh, scene);
}

} /* end namespace detail */

} /* end namespace python */

} /* end namespace modmesh */

PYBIND11_EMBEDDED_MODULE(_modmesh_view, mod)
{
    using namespace modmesh;
    using namespace modmesh::python;
    namespace py = pybind11;

    mod
        .def("show", &modmesh::python::detail::update_appmesh, py::arg("mesh"))
        //
        ;

    WrapR3DWidget::commit(mod, "R3DWidget", "R3DWidget");
    WrapRApplication::commit(mod, "RApplication", "RApplication");

    mod.attr("app") = py::cast(RApplication::instance());
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
