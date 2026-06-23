#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/DrawTool.hpp>
#include <solvcon/pilot/R2DWidget.hpp>
#include <solvcon/pilot/R3DWidget.hpp>
#include <solvcon/pilot/RAction.hpp>
#include <solvcon/pilot/RPythonConsoleDockWidget.hpp>

#include <vector>

#include <QApplication>
#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <Qt>

namespace solvcon
{

class RManager
    : public QObject
{
    Q_OBJECT

public:

    ~RManager() override;

    RManager & setUp();

    static RManager & instance();

    QCoreApplication * core() { return m_core.get(); }

    R3DWidget * add3DWidget();
    R2DWidget * add2DWidget();
    R3DWidget * currentR3DWidget();
    R2DWidget * currentR2DWidget();
    std::vector<R2DWidget *> list2DWidgets();

    /// Name of the active canvas drawing tool.
    std::string drawTool() const { return m_draw_tool; }

    /// Select the active drawing tool and apply it to the focused 2D canvas.
    void setDrawTool(std::string const & name);

    RPythonConsoleDockWidget * pycon() { return m_pycon; }

    QMainWindow * mainWindow() { return m_mainWindow; }

    QMdiArea * mdiArea() { return m_mdiArea; }

    template <typename... Args>
    QMdiSubWindow * addSubWindow(Args &&... args);

    QMenu * fileMenu() { return m_fileMenu; }
    QMenu * editMenu() { return m_editMenu; }
    QMenu * viewMenu() { return m_viewMenu; }
    QMenu * oneMenu() { return m_oneMenu; }
    QMenu * meshMenu() { return m_meshMenu; }
    QMenu * canvasMenu() { return m_canvasMenu; }
    QMenu * profilingMenu() { return m_profilingMenu; }
    QMenu * windowMenu() { return m_windowMenu; }

    void quit() { m_core->quit(); }

    /// Only call reset() when the program is to be stopped.
    void reset();

    void toggleConsole();

private:

    RManager();

    void setUpConsole();
    void setUpCentral();
    void setUpMenu();

    /// Push the active draw tool onto the focused 2D canvas, if any. A
    /// no-op when the focused subwindow is not a 2D canvas.
    void applyDrawTool();

    void setUpEditMenuItems() const;
    void setUpCameraControllersMenuItems() const;
    void setUpCameraMovementMenuItems() const;

    /// Undo or redo the most recent shape change on the focused 2D canvas,
    /// then repaint it. A no-op when no 2D canvas is focused.
    void undoCanvas() const;
    void redoCanvas() const;

    std::function<void()> createCameraMovementItemHandler(const std::function<void(CameraInputState &)> &) const;

    bool m_already_setup = false;

    std::unique_ptr<QCoreApplication> m_core = nullptr;

    QMainWindow * m_mainWindow = nullptr;

    QMenu * m_fileMenu = nullptr;
    QMenu * m_editMenu = nullptr;
    QMenu * m_viewMenu = nullptr;
    QMenu * m_oneMenu = nullptr;
    QMenu * m_meshMenu = nullptr;
    QMenu * m_canvasMenu = nullptr;
    QMenu * m_profilingMenu = nullptr;
    QMenu * m_windowMenu = nullptr;

    RPythonConsoleDockWidget * m_pycon = nullptr;
    QMdiArea * m_mdiArea = nullptr;

    /// Active canvas drawing tool, shared by every 2D canvas. Starts on
    /// the default tool (pan navigation).
    std::string m_draw_tool = default_draw_tool_name();
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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
