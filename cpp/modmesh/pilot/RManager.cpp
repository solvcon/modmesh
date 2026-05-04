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

#include <modmesh/pilot/RManager.hpp> // Must be the first include.

#include <modmesh/pilot/RAction.hpp>
#include <modmesh/pilot/RMenu.hpp>
#include <Qt>
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
    m_core.reset(QApplication::instance());
    static int argc = 1;
    static char exename[] = "pilot";
    static char * argv[] = {exename};
    if (!m_core)
    {
        m_core.reset(new QApplication(argc, argv));
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

void RManager::reset()
{
    // Nullify all child pointers before destroying QApplication.
    // Qt's parent-child mechanism will delete widget children; we only
    // need to ensure our raw pointers don't dangle after the call.
    m_already_setup = false;
    m_menus.clear();
    m_menuBar = nullptr;
    m_pycon = nullptr;
    m_mdiArea = nullptr;
    m_mainWindow = nullptr;
    m_core.reset();
}

RManager::~RManager()
{
    reset();
}

RMenu * RManager::addMenu(std::string const & title)
{
    if (!m_menuBar)
    {
        return nullptr;
    }
    auto * menu = new RMenu(QString::fromStdString(title), m_menuBar);
    m_menuBar->addMenu(menu);
    m_menus[title] = menu;
    return menu;
}

void RManager::addViewMenuCameraItems()
{
    RMenu * view_menu = viewMenu();
    if (view_menu)
    {
        setUpCameraControllersMenuItems(view_menu);
        setUpCameraMovementMenuItems(view_menu);
    }
}

RMenu * RManager::findMenu(std::string const & title) const
{
    auto it = m_menus.find(title);
    return (it != m_menus.end()) ? it->second : nullptr;
}

R3DWidget * RManager::add3DWidget()
{
    R3DWidget * viewer = nullptr;
    if (m_mdiArea)
    {
        viewer = new R3DWidget(/*window*/ nullptr, /*scene*/ nullptr, /*parent*/ m_mdiArea);
        viewer->setWindowTitle("3D viewer");
        viewer->show();
        auto * subwin = this->addSubWindow(viewer);
        subwin->resize(400, 300);
    }
    return viewer;
}

void RManager::toggleConsole()
{
    if (m_pycon)
    {
        if (m_pycon->isVisible())
        {
            m_pycon->hide();
        }
        else
        {
            m_pycon->show();
        }
    }
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
    m_menuBar = new QMenuBar(nullptr);
    m_mainWindow->setMenuBar(m_menuBar);
}

void RManager::setUpCameraControllersMenuItems(RMenu * view_menu) const
{
    auto * use_orbit_camera = new RAction(
        QString("Use Orbit Camera Controller"),
        QString("Use Oribt Camera Controller"),
        [this]()
        {
            qDebug() << "Use Orbit Camera Controller (menu demo)";
            for (auto subwin : m_mdiArea->subWindowList())
            {
                auto * viewer = dynamic_cast<R3DWidget *>(subwin->widget());

                if (viewer == nullptr)
                    continue;

                viewer->scene()->setOrbitCameraController();
                viewer->scene()->controller()->setCamera(viewer->camera());
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
                auto * viewer = dynamic_cast<R3DWidget *>(subwin->widget());

                if (viewer == nullptr)
                    continue;

                viewer->scene()->setFirstPersonCameraController();
                viewer->scene()->controller()->setCamera(viewer->camera());
            }
        });

    auto * cameraGroup = new QActionGroup(m_mainWindow);
    cameraGroup->addAction(use_orbit_camera);
    cameraGroup->addAction(use_fps_camera);

    use_orbit_camera->setCheckable(true);
    use_fps_camera->setCheckable(true);
    use_orbit_camera->setChecked(true);

    view_menu->addAction(use_orbit_camera);
    view_menu->addAction(use_fps_camera);
}

void RManager::setUpCameraMovementMenuItems(RMenu * view_menu) const
{
    auto * reset_camera = new RAction(
        QString("Reset (esc)"),
        QString("Reset (esc)"),
        [this]()
        {
            const auto * subwin = m_mdiArea->currentSubWindow();
            if (subwin == nullptr)
                return;

            auto * viewer = dynamic_cast<R3DWidget *>(subwin->widget());
            if (viewer == nullptr || viewer->camera() == nullptr)
                return;

            viewer->cameraController()->reset();
        });

    auto * move_camera_up = new RAction(
        QString("Move camera up (W/UP)"),
        QString("Move camera up (W/UP)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.tyAxisValue = 1.0; }));

    auto * move_camera_down = new RAction(
        QString("Move camera down (S/DOWN)"),
        QString("Move camera down (S/DOWN)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.tyAxisValue = -1.0; }));

    auto * move_camera_right = new RAction(
        QString("Move camera right (D/RIGHT)"),
        QString("Move camera right (D/RIGHT)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.txAxisValue = 1.0; }));

    auto * move_camera_left = new RAction(
        QString("Move camera left (A/LEFT)"),
        QString("Move camera left (A/LEFT)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.txAxisValue = -1.0; }));

    auto * move_camera_forward = new RAction(
        QString("Move camera forward (Ctrl+W/UP)"),
        QString("Move camera forward (Ctrl+W/UP)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.tzAxisValue = 1.0; }));

    auto * move_camera_backward = new RAction(
        QString("Move camera backward (Ctrl+S/DOWN)"),
        QString("Move camera backward (Ctrl+S/DOWN)"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.tzAxisValue = -1.0; }));

    auto * rotate_camera_positive_yaw = new RAction(
        QString("Rotate camera positive yaw"),
        QString("Rotate camera positive yaw"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.rxAxisValue = 1.0; }));

    auto * rotate_camera_negative_yaw = new RAction(
        QString("Rotate camera negative yaw"),
        QString("Rotate camera negative yaw"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.rxAxisValue = -1.0; }));

    auto * rotate_camera_positive_pitch = new RAction(
        QString("Rotate camera positive pitch"),
        QString("Rotate camera positive pitch"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.ryAxisValue = 1.0; }));

    auto * rotate_camera_negative_pitch = new RAction(
        QString("Rotate camera negative pitch"),
        QString("Rotate camera negative pitch"),
        createCameraMovementItemHandler([](CameraInputState & input)
                                        { input.ryAxisValue = -1.0; }));

    reset_camera->setShortcut(QKeySequence(Qt::Key_Escape));
    reset_camera->setShortcutContext(Qt::WidgetShortcut);

    auto * cameraMoveSubmenu = view_menu->addMenu("Camera move");
    cameraMoveSubmenu->addAction(reset_camera);
    cameraMoveSubmenu->addAction(move_camera_up);
    cameraMoveSubmenu->addAction(move_camera_down);
    cameraMoveSubmenu->addAction(move_camera_right);
    cameraMoveSubmenu->addAction(move_camera_left);
    cameraMoveSubmenu->addAction(move_camera_forward);
    cameraMoveSubmenu->addAction(move_camera_backward);
    cameraMoveSubmenu->addAction(rotate_camera_positive_yaw);
    cameraMoveSubmenu->addAction(rotate_camera_negative_yaw);
    cameraMoveSubmenu->addAction(rotate_camera_positive_pitch);
    cameraMoveSubmenu->addAction(rotate_camera_negative_pitch);
}

std::function<void()> RManager::createCameraMovementItemHandler(
    const std::function<void(CameraInputState &)> & func) const
{
    return [this, func]()
    {
        const auto * subwin = m_mdiArea->currentSubWindow();
        if (subwin == nullptr)
            return;

        auto * viewer = dynamic_cast<R3DWidget *>(subwin->widget());
        if (viewer == nullptr || viewer->camera() == nullptr)
            return;

        const auto controllerType = viewer->cameraController()->getType();
        CameraInputState input{};

        func(input);

        if (input.rxAxisValue != 0.f || input.ryAxisValue != 0.f)
        {
            if (controllerType == CameraControllerType::Orbit)
            {
                constexpr float orbitRotationSpeed = 5.0f;

                input.rxAxisValue *= orbitRotationSpeed;
                input.ryAxisValue *= orbitRotationSpeed;
                input.rightMouseButtonActive = true;
            }
            else if (controllerType == CameraControllerType::FirstPerson)
            {
                input.leftMouseButtonActive = true;
            }
        }

        viewer->cameraController()->moveCamera(input, 0.01);
    };
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
