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

#include <modmesh/view/common_detail.hpp> // Must be the first include.

#include <Qt>
#include <QWidget>
#include <Qt3DWindow>

#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>

#include <QAbstractCameraController>
#include <QOrbitCameraController>
#include <QFirstPersonCameraController>

#include <QResizeEvent>

namespace modmesh
{

class RFirstPersonCameraController
    : public Qt3DExtras::QFirstPersonCameraController
{
    using Qt3DExtras::QFirstPersonCameraController::QFirstPersonCameraController;
}; /* end class RFirstPersonCameraController */

class ROrbitCameraController
    : public Qt3DExtras::QOrbitCameraController
{
    using Qt3DExtras::QOrbitCameraController::QOrbitCameraController;
}; /* end class ROrbitCameraController */

class RScene
    : public Qt3DCore::QEntity
{

public:

    RScene(Qt3DCore::QNode * parent = nullptr)
        : Qt3DCore::QEntity(parent)
        , m_controller(new ROrbitCameraController(this))
    {
    }

    Qt3DExtras::QAbstractCameraController const * controller() const { return m_controller; }
    Qt3DExtras::QAbstractCameraController * controller() { return m_controller; }

    void setCameraController(Qt3DExtras::QAbstractCameraController * controller)
    {
        m_controller->deleteLater();
        m_controller = controller;
    }

    void setOrbitCameraController() { setCameraController(new ROrbitCameraController(this)); }

    void setFirstPersonCameraController() { setCameraController(new RFirstPersonCameraController(this)); }

private:

    Qt3DExtras::QAbstractCameraController * m_controller = nullptr;

}; /* end class RScene */

class R3DWidget
    : public QWidget
{

public:

    R3DWidget(
        Qt3DExtras::Qt3DWindow * window = nullptr,
        RScene * scene = nullptr,
        QWidget * parent = nullptr,
        Qt::WindowFlags f = Qt::WindowFlags());

    template <typename... Args>
    void resize(Args &&... args);

    void resizeEvent(QResizeEvent * event);

    Qt3DExtras::Qt3DWindow * view() { return m_view; }
    RScene * scene() { return m_scene; }
    Qt3DRender::QCamera * camera() { return m_view->camera(); }

    QPixmap grabPixmap() const { return m_view->screen()->grabWindow(m_view->winId()); }

private:

    Qt3DExtras::Qt3DWindow * m_view = nullptr;
    RScene * m_scene = nullptr;
    QWidget * m_container = nullptr;

}; /* end class R3DWidget */

template <typename... Args>
void R3DWidget::resize(Args &&... args)
{
    QWidget::resize(std::forward<Args>(args)...);
    m_view->resize(std::forward<Args>(args)...);
    m_container->resize(std::forward<Args>(args)...);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
