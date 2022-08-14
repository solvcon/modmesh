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

#include <modmesh/view/RApplication.hpp> // Must be the first include.

#include <modmesh/view/RMainWindow.hpp>

#include <vector>

#include <QActionGroup>

namespace modmesh
{

RApplication::RApplication(int & argc, char ** argv)
    : QApplication(argc, argv)
    , m_main(new RMainWindow)
{
    /* TODO: parse arguments */

    // Set up menu.
    /* TODO: Menu setup may be moved to Python */
    auto * menuBar = new RMenuBar();
    auto * fileMenu = new RMenu(QString("File"));
    auto * appMenu = new RMenu(QString("App"));
    auto * cameraMenu = new RMenu(QString("Camera"));

    auto * newFileAction = new RAction(
        QString("New file"),
        QString("Create new file"),
        []()
        {
            qDebug() << "This is only a demo: Create new file!";
        });

    auto * newMenu = new RMenu(QString("New"));
    newMenu->addAction(newFileAction);
    fileMenu->addMenu(newMenu);

#ifndef Q_OS_MACOS
    // Qt for mac merges "quit" or "exit" with the default quit item in the
    // system menu:
    // https://doc.qt.io/qt-6/qmenubar.html#qmenubar-as-a-global-menu-bar
    RAction * exitAction = new RAction(
        QString("Exit"),
        QString("Exit the application"),
        quit);
    fileMenu->addAction(exitAction);
#endif

    menuBar->addMenu(fileMenu);

    auto * app_sample_mesh = new RAction(
        QString("Load sample_mesh"),
        QString("Load sample_mesh"),
        []()
        {
            namespace py = pybind11;
            py::module_ appmod = py::module_::import("modmesh.app.sample_mesh");
            appmod.reload();
            appmod.attr("load_app")();
        });
    appMenu->addAction(app_sample_mesh);

    auto * app_linear_wave = new RAction(
        QString("Load linear_wave"),
        QString("Load linear_wave"),
        []()
        {
            namespace py = pybind11;
            py::module_ appmod = py::module_::import("modmesh.app.linear_wave");
            appmod.reload();
            appmod.attr("load_app")();
        });
    appMenu->addAction(app_linear_wave);

    auto * app_euler1d = new RAction(
        QString("Load euler1d"),
        QString("Load euler1d"),
        []()
        {
            namespace py = pybind11;
            py::module_ appmod = py::module_::import("modmesh.app.euler1d");
            appmod.reload();
            appmod.attr("load_app")();
        });
    appMenu->addAction(app_euler1d);

    auto * app_bad_euler1d = new RAction(
        QString("Load bad_euler1d"),
        QString("Load bad_euler1d"),
        []()
        {
            namespace py = pybind11;
            py::module_ appmod = py::module_::import("modmesh.app.bad_euler1d");
            appmod.reload();
            appmod.attr("load_app")();
        });
    appMenu->addAction(app_bad_euler1d);

    menuBar->addMenu(appMenu);

    auto * use_orbit_camera = new RAction(
        QString("Use Oribt Camera Controller"),
        QString("Use Oribt Camera Controller"),
        [&]()
        {
            qDebug() << "Use Orbit Camera Controller";
            auto * viewer = this->m_main->viewer();
            viewer->scene()->setOrbitCameraController();
            viewer->scene()->controller()->setCamera(viewer->camera());
        });

    auto * use_fps_camera = new RAction(
        QString("Use First Person Camera Controller"),
        QString("Use First Person Camera Controller"),
        [&]()
        {
            qDebug() << "Use First Person Camera Controller";
            auto * viewer = this->m_main->viewer();
            viewer->scene()->setFirstPersonCameraController();
            viewer->scene()->controller()->setCamera(viewer->camera());
        });

    auto * cameraGroup = new QActionGroup(this);
    cameraGroup->addAction(use_orbit_camera);
    cameraGroup->addAction(use_fps_camera);
    use_orbit_camera->setCheckable(true);
    use_fps_camera->setCheckable(true);
    use_orbit_camera->setChecked(true);

    cameraMenu->addAction(use_orbit_camera);
    cameraMenu->addAction(use_fps_camera);
    menuBar->addMenu(cameraMenu);

    m_main->setMenuBar(menuBar);

    // Setup main window
    m_main->setWindowTitle(m_title);
    m_main->setWindowIcon(QIcon(m_iconFilePath));

    // Show main window.
    m_main->show();
}

RApplication::~RApplication()
{
    // Shuts down the interpreter when the application stops.
    python::Interpreter::instance().finalize();
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
