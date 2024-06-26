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

#include <modmesh/view/RManager.hpp> // Must be the first include.

#include <vector>

#include <modmesh/view/RAction.hpp>
#include <modmesh/view/RParameter.hpp>
#include <Qt>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QActionGroup>

namespace modmesh
{

RManager & RManager::instance()
{
    static RManager ret;
    return ret;
}

RManager::RManager()
    : QObject()
{
    m_core = QApplication::instance();
    static int argc = 1;
    static char exename[] = "viewer";
    static char * argv[] = {exename};
    if (nullptr == m_core)
    {
        m_core = new QApplication(argc, argv);
    }

    m_mainWindow = new QMainWindow;
    m_mainWindow->setWindowIcon(QIcon(QString(":/icon.ico")));
    // Do not call setUp() from the constructor.  Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.
}

RManager & RManager::setUp()
{
    if (!m_already_setup)
    {
        this->setUpConsole();
        this->setUpCentral();
        this->setUpMenu();

        m_already_setup = true;
    }
    return *this;
}

RManager::~RManager()
{
}

R3DWidget * RManager::add3DWidget()
{
    R3DWidget * viewer = nullptr;
    if (m_mdiArea)
    {
        viewer = new R3DWidget();
        viewer->setWindowTitle("3D viewer");
        viewer->show();
        auto * subwin = this->addSubWindow(viewer);
        subwin->resize(400, 300);
    }
    return viewer;
}

void RManager::setUpConsole()
{
    m_pycon = new RPythonConsoleDockWidget(QString("Console"), m_mainWindow);
    m_pycon->setAllowedAreas(Qt::AllDockWidgetAreas);
    m_mainWindow->addDockWidget(Qt::BottomDockWidgetArea, m_pycon);
}

void RManager::setUpCentral()
{
    m_mdiArea = new QMdiArea(m_mainWindow);
    m_mdiArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_mdiArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_mainWindow->setCentralWidget(m_mdiArea);
}

void RManager::setUpMenu()
{
    m_mainWindow->setMenuBar(new QMenuBar(nullptr));
    // NOTE: All menus need to be populated or Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.

    m_fileMenu = m_mainWindow->menuBar()->addMenu(QString("File"));
    {
        m_fileMenu->addAction(
            new RAction(
                QString("New file"),
                QString("Create new file"),
                []()
                {
                    qDebug() << "This is only a demo: Create new file!";
                }));

        m_fileMenu->addAction(new RAction(
            QString("Parameters"),
            QString("Runtime parameters"),
            []()
            {
                static int64_t int64V = 5566;
                static double doubleV = 77.88;
                auto params = createParameters();
                addParam(params, "global.a.b.int64_foo", &int64V);
                addParam(params, "global.a.b.double_bar", &doubleV);
                openParameterView(params);
            }));

#ifndef Q_OS_MACOS
        // Qt for mac merges "quit" or "exit" with the default quit item in the
        // system menu:
        // https://doc.qt.io/qt-6/qmenubar.html#qmenubar-as-a-global-menu-bar
        m_fileMenu->addAction(new RAction(
            QString("Exit"),
            QString("Exit the application"),
            []()
            {
                RManager::instance().quit();
            }));
#endif
    }

    m_viewMenu = m_mainWindow->menuBar()->addMenu(QString("View"));
    {
        auto * use_orbit_camera = new RAction(
            QString("Use Orbit Camera Controller"),
            QString("Use Oribt Camera Controller"),
            [this]()
            {
                qDebug() << "Use Orbit Camera Controller (menu demo)";
                for (auto subwin : m_mdiArea->subWindowList())
                {
                    try
                    {
                        R3DWidget * viewer = dynamic_cast<R3DWidget *>(subwin->widget());
                        viewer->scene()->setOrbitCameraController();
                        viewer->scene()->controller()->setCamera(viewer->camera());
                    }
                    catch (std::bad_cast & e)
                    {
                        std::cerr << e.what() << std::endl;
                    }
                }
            });

        auto * use_fps_camera = new RAction(
            QString("Use First Person Camera Controller"),
            QString("Use First Person Camera Controller"),
            [this]()
            {
                qDebug() << "Use First Person Camera Controller (menu demo)";
                for (auto subwin : m_mdiArea->subWindowList())
                {
                    try
                    {
                        R3DWidget * viewer = dynamic_cast<R3DWidget *>(subwin->widget());
                        viewer->scene()->setFirstPersonCameraController();
                        viewer->scene()->controller()->setCamera(viewer->camera());
                    }
                    catch (std::bad_cast & e)
                    {
                        std::cerr << e.what() << std::endl;
                    }
                }
            });

        auto * reset_camera = new RAction(
            QString("Reset (esc)"),
            QString("Reset (esc)"),
            [this]()
            {
                // implement resetting camera contorl
                qDebug() << "Reset to initial status.";
                for (auto subwin : m_mdiArea->subWindowList())
                {
                    try
                    {
                        R3DWidget * viewer = dynamic_cast<R3DWidget *>(subwin->widget());
                        Qt3DRender::QCamera * camera = viewer->camera();
                        if (camera)
                        {
                            viewer->resetCamera(camera);
                        }
                    }
                    catch (std::bad_cast & e)
                    {
                        std::cerr << e.what() << std::endl;
                    }
                }
            });

        auto * cameraGroup = new QActionGroup(m_mainWindow);
        cameraGroup->addAction(use_orbit_camera);
        cameraGroup->addAction(use_fps_camera);
        cameraGroup->addAction(reset_camera);

        use_orbit_camera->setCheckable(true);
        use_fps_camera->setCheckable(true);
        use_orbit_camera->setChecked(true);

        reset_camera->setShortcut(QKeySequence(Qt::Key_Escape));

        m_viewMenu->addAction(use_orbit_camera);
        m_viewMenu->addAction(use_fps_camera);
        m_viewMenu->addAction(reset_camera);
    }

    m_oneMenu = m_mainWindow->menuBar()->addMenu(QString("One"));
    m_oneMenu->addAction(new RAppAction(
        QString("Euler solver"),
        QString("One-dimensional shock-tube problem with Euler solver"),
        QString("modmesh.app.euler1d")));

    m_meshMenu = m_mainWindow->menuBar()->addMenu(QString("Mesh"));
    m_meshMenu->addAction(new RPythonAction(
        QString("Sample: mesh of a triangle (2D)"),
        QString("Create a very simple sample mesh of a triangle"),
        QString("modmesh.gui.sample_mesh.mesh_triangle")));
    m_meshMenu->addAction(new RPythonAction(
        QString("Sample: mesh of a tetrahedron (3D)"),
        QString("Create a very simple sample mesh of a tetrahedron"),
        QString("modmesh.gui.sample_mesh.mesh_tetrahedron")));
    m_meshMenu->addAction(new RPythonAction(
        QString("Sample: mesh of \"solvcon\" text in 2D"),
        QString("Create a sample mesh drawing a text string of \"solvcon\""),
        QString("modmesh.gui.sample_mesh.mesh_solvcon_2dtext")));
    m_meshMenu->addAction(new RPythonAction(
        QString("Sample: 3D mesh of mixed elements"),
        QString("Create a very simple sample mesh of mixed elements in 3D"),
        QString("modmesh.gui.sample_mesh.mesh_3dmix")));
    m_meshMenu->addAction(new RPythonAction(
        QString("Sample: NACA 4-digit"),
        QString("Draw a NACA 4-digit airfoil"),
        QString("modmesh.gui.naca.runmain")));

    m_addonMenu = m_mainWindow->menuBar()->addMenu(QString("Addon"));
    this->addApplication(QString("sample_mesh"));
    this->addApplication(QString("linear_wave"));
    this->addApplication(QString("bad_euler1d"));

    m_windowMenu = m_mainWindow->menuBar()->addMenu(QString("Window"));
    m_windowMenu->addAction(
        new RAction(
            QString("(empty)"),
            QString("(empty)"),
            []() {}));
}

void RManager::clearApplications()
{
    for (QAction * a : this->m_addonMenu->actions())
    {
        auto * p = dynamic_cast<RAppAction *>(a);
        if (nullptr != p)
        {
            this->m_addonMenu->removeAction(a);
        }
    }
}

void RManager::addApplication(QString const & name)
{
    m_addonMenu->addAction(new RAppAction(
        QString("Load ") + name,
        QString("Load ") + name,
        QString("modmesh.app.") + name));
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
