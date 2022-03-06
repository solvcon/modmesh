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

#include <modmesh/python/python.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>
#include <modmesh/viewer/viewer.hpp>

#include <pybind11/embed.h>

#include <Qt>
#include <QWidget>
#include <QApplication>
#include <QMainWindow>

#include <qt3dwindow.h>

std::shared_ptr<modmesh::StaticMesh2d> make_3triangles()
{
    auto mh = modmesh::StaticMesh2d::construct(/*nnode*/ 4, /*nface*/ 0, /*ncell*/ 3);

    mh->ndcrd(0, 0) = 0;
    mh->ndcrd(0, 1) = 0;
    mh->ndcrd(1, 0) = -1;
    mh->ndcrd(1, 1) = -1;
    mh->ndcrd(2, 0) = 1;
    mh->ndcrd(2, 1) = -1;
    mh->ndcrd(3, 0) = 0;
    mh->ndcrd(3, 1) = 1;

    std::fill(mh->cltpn().begin(), mh->cltpn().end(), modmesh::CellType::TRIANGLE);

    mh->clnds(0, 0) = 3;
    mh->clnds(0, 1) = 0;
    mh->clnds(0, 2) = 1;
    mh->clnds(0, 3) = 2;
    mh->clnds(1, 0) = 3;
    mh->clnds(1, 1) = 0;
    mh->clnds(1, 2) = 2;
    mh->clnds(1, 3) = 3;
    mh->clnds(2, 0) = 3;
    mh->clnds(2, 1) = 0;
    mh->clnds(2, 2) = 3;
    mh->clnds(2, 3) = 1;

    mh->build_interior(/*do_metric*/ true);
    mh->build_boundary();
    mh->build_ghost();

    return mh;
}

int main(int argc, char ** argv)
{
    /*
     * TODO: Sequence of application startup:
     *   1. Parsing arguments and parameters.
     *   2. Initialize application globals.
     *   3. Initialize GUI globals.
     *   4. Set up GUI windowing.
     */

    using namespace modmesh;
    namespace py = pybind11;

    py::scoped_interpreter interpreter_guard{};

    // Load the Python extension module.
    std::cout << "Loading modmesh._modmesh ... ";
    try
    {
        py::module_::import("modmesh._modmesh");
    }
    catch (const py::error_already_set & e)
    {
        if (std::string::npos == std::string(e.what()).find("ModuleNotFoundError"))
        {
            throw;
        }
        else
        {
            std::cout << "fails";
        }
    }
    std::cout << "succeeds" << std::endl;

    // Start application with GUI.
    QApplication app(argc, argv);

    // Create and set up main 3D view.
    auto * view = new Qt3DExtras::Qt3DWindow;

    {
        // Set up the camera.
        Qt3DRender::QCamera * camera = view->camera();
        camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
        camera->setPosition(QVector3D(0, 0, 40.0f));
        camera->setViewCenter(QVector3D(0, 0, 0));
    }

    // Create and set up the root scene.
    RScene scene;
    view->setRootEntity(scene.ptr());

    {
        // Set up the camera control.
        auto * control = scene.camera_controller();
        control->setCamera(view->camera());
        control->setLinearSpeed(50.0f);
        control->setLookSpeed(180.0f);

        // Set the mesh to the scene.
        RStaticMesh<2> rmh(make_3triangles());
        rmh->setParent(scene.ptr());
    }

    auto * widget = new QWidget();
    auto * container = widget->createWindowContainer(view, widget, Qt::Widget);
    container->resize(400, 400);
    widget->resize(400, 400);

    QMainWindow window;
    window.setCentralWidget(widget);
    window.resize(400, 400);
    window.show();
    return app.exec();
}
