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

#include <modmesh/view/RMainWindow.hpp> // Must be the first include.

#include <modmesh/view/R3DWidget.hpp>

#include <modmesh/view/RAction.hpp>
#include <modmesh/view/RApplication.hpp>
#include <Qt>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QActionGroup>

namespace modmesh
{

RMainWindow::RMainWindow()
    : QMainWindow()
{
    // Do not call setUp() from the constructor.  Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.
}

void RMainWindow::setUp()
{
    if (m_already_setup)
    {
        return;
    }

    this->setWindowIcon(QIcon(QString(":/icon.ico")));

    this->setMenuBar(new QMenuBar(nullptr));
    m_fileMenu = this->menuBar()->addMenu(QString("File"));
    m_appMenu = this->menuBar()->addMenu(QString("App"));
    m_cameraMenu = this->menuBar()->addMenu(QString("Camera"));
    // NOTE: All menus need to be populated or Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.

    {
        auto * action = new RAction(
            QString("New file"),
            QString("Create new file"),
            []()
            {
                qDebug() << "This is only a demo: Create new file!";
            });
        m_fileMenu->addAction(action);
    }

#ifndef Q_OS_MACOS
    {
        // Qt for mac merges "quit" or "exit" with the default quit item in the
        // system menu:
        // https://doc.qt.io/qt-6/qmenubar.html#qmenubar-as-a-global-menu-bar
        auto * action = new RAction(
            QString("Exit"),
            QString("Exit the application"),
            []()
            {
                RApplication::instance()->quit();
            });
        m_fileMenu->addAction(action);
    }
#endif

    {
        this->addApplication(QString("sample_mesh"));
        this->addApplication(QString("euler1d"));
        this->addApplication(QString("linear_wave"));
        this->addApplication(QString("bad_euler1d"));
    }

    {
        auto * use_orbit_camera = new RAction(
            QString("Use Oribt Camera Controller"),
            QString("Use Oribt Camera Controller"),
            [this]()
            {
                qDebug() << "Use Orbit Camera Controller";
                auto * viewer = this->viewer();
                viewer->scene()->setOrbitCameraController();
                viewer->scene()->controller()->setCamera(viewer->camera());
            });

        auto * use_fps_camera = new RAction(
            QString("Use First Person Camera Controller"),
            QString("Use First Person Camera Controller"),
            [this]()
            {
                qDebug() << "Use First Person Camera Controller";
                auto * viewer = this->viewer();
                viewer->scene()->setFirstPersonCameraController();
                viewer->scene()->controller()->setCamera(viewer->camera());
            });

        auto * cameraGroup = new QActionGroup(this);
        cameraGroup->addAction(use_orbit_camera);
        cameraGroup->addAction(use_fps_camera);
        use_orbit_camera->setCheckable(true);
        use_fps_camera->setCheckable(true);
        use_orbit_camera->setChecked(true);

        m_cameraMenu->addAction(use_orbit_camera);
        m_cameraMenu->addAction(use_fps_camera);
    }

    m_pycon = new RPythonConsoleDockWidget(QString("Console"), this);
    m_pycon->setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::BottomDockWidgetArea, m_pycon);

    m_viewer = new R3DWidget();
    setCentralWidget(m_viewer);

    m_already_setup = true;
}

void RMainWindow::clearApplications()
{
    for (QAction * a : this->m_appMenu->actions())
    {
        auto * p = dynamic_cast<RAppAction *>(a);
        if (nullptr != p)
        {
            this->m_appMenu->removeAction(a);
        }
    }
}

void RMainWindow::addApplication(QString const & name)
{
    m_appMenu->addAction(new RAppAction(
        QString("Load ") + name,
        QString("Load ") + name,
        QString("modmesh.app.") + name));
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
