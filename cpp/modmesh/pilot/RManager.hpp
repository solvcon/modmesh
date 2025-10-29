#pragma once

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

#include <modmesh/pilot/common_detail.hpp> // Must be the first include.

#include <modmesh/pilot/RPythonConsoleDockWidget.hpp>
#include <modmesh/pilot/RVisionDockWidget.hpp>
#include <modmesh/pilot/R3DWidget.hpp>
#include <modmesh/pilot/RAction.hpp>

#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <Qt>

namespace modmesh
{

class RManager
    : public QObject
{
    Q_OBJECT

public:

    ~RManager() override;

    RManager & setUp();

    static RManager & instance();

    QCoreApplication * core() { return m_core; }

    R3DWidget * add3DWidget();

    RPythonConsoleDockWidget * pycon() { return m_pycon; }
    RVisionDockWidget * vision() { return m_vision; }

    QMainWindow * mainWindow() { return m_mainWindow; }

    template <typename... Args>
    QMdiSubWindow * addSubWindow(Args &&... args);

    QMenu * fileMenu() { return m_fileMenu; }
    QMenu * viewMenu() { return m_viewMenu; }
    QMenu * oneMenu() { return m_oneMenu; }
    QMenu * meshMenu() { return m_meshMenu; }
    QMenu * profilingMenu() { return m_profilingMenu; }
    QMenu * windowMenu() { return m_windowMenu; }

    void quit() { m_core->quit(); }

    void toggleConsole();
    void toggleVision();

private:

    RManager();

    void setUpConsole();
    void setUpVision();
    void setUpCentral();
    void setUpMenu();

    void setUpCameraControllersMenuItems() const;
    void setUpCameraMovementMenuItems() const;

    std::function<void()> createCameraMovementItemHandler(const std::function<void(CameraInputState &)> &) const;

    bool m_already_setup = false;

    QCoreApplication * m_core = nullptr;

    QMainWindow * m_mainWindow = nullptr;

    QMenu * m_fileMenu = nullptr;
    QMenu * m_viewMenu = nullptr;
    QMenu * m_oneMenu = nullptr;
    QMenu * m_meshMenu = nullptr;
    QMenu * m_profilingMenu = nullptr;
    QMenu * m_windowMenu = nullptr;

    RPythonConsoleDockWidget * m_pycon = nullptr;
    RVisionDockWidget * m_vision = nullptr;
    QMdiArea * m_mdiArea = nullptr;
}; /* end class RManager */

template <typename... Args>
QMdiSubWindow * RManager::addSubWindow(Args &&... args)
{
    QMdiSubWindow * subwin = nullptr;
    if (m_mdiArea)
    {
        subwin = m_mdiArea->addSubWindow(std::forward<Args>(args)...);
        subwin->show();
        m_mdiArea->setActiveSubWindow(subwin);
    }
    return subwin;
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
