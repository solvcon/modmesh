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

#include <modmesh/python/wrapper/view/view.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

#include <modmesh/view/view.hpp>

#include <QClipboard>

namespace modmesh
{

namespace python
{

namespace detail
{

static void show_mark()
{
    RScene * scene = RApplication::instance()->main()->viewer()->scene();
    for (Qt3DCore::QNode * child : scene->childNodes())
    {
        if (typeid(*child) == typeid(RAxisMark))
        {
            child->deleteLater();
        }
    }
    new RAxisMark(scene);
}

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
        .def("showMark", &modmesh::python::detail::show_mark)
        .def(
            "clipImage",
            []()
            {
                R3DWidget * viewer = RApplication::instance()->main()->viewer();
                QClipboard * clipboard = QGuiApplication::clipboard();
                clipboard->setPixmap(viewer->grabPixmap());
            })
        .def(
            "saveImage",
            [](std::string const & filename)
            {
                R3DWidget * viewer = RApplication::instance()->main()->viewer();
                viewer->grabPixmap().save(filename.c_str());
            },
            py::arg("filename"))
        //
        ;

    wrap_view(mod);

    mod.attr("app") = py::cast(RApplication::instance());
}

int main(int argc, char ** argv)
{
    using namespace modmesh;

    RApplication app(argc, argv);
    app.main()->resize(1000, 600);

    python::detail::show_mark();

    return app.exec();
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: