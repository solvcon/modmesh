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

#include <modmesh/pilot/RWorld.hpp>

#include <Qt>
#include <QWidget>
#include <Qt3DWindow>

#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>

#include <modmesh/pilot/RCameraController.hpp>

#include <QResizeEvent>

namespace modmesh
{

class RScene
    : public Qt3DCore::QEntity
{
    Q_OBJECT

public:

    explicit RScene(QNode * parent = nullptr)
        : QEntity(parent)
    {
        m_controller = new ROrbitCameraController(this);
        m_min_pt = QVector3D(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        m_max_pt = QVector3D(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    }

    RCameraController * controller() const { return m_controller; }

    void setCameraController(RCameraController * controller)
    {
        m_controller->deleteLater();
        m_controller = controller;
    }

    void setOrbitCameraController() { setCameraController(new ROrbitCameraController(this)); }

    void setFirstPersonCameraController() { setCameraController(new RFirstPersonCameraController(this)); }

    void updateBoundingBox(QVector3D const & buttom_left, QVector3D const & top_right)
    {
        float bbox_min_x = std::min(buttom_left.x(), m_min_pt.x());
        float bbox_min_y = std::min(buttom_left.y(), m_min_pt.y());
        float bbox_min_z = std::min(buttom_left.z(), m_min_pt.z());
        float bbox_max_x = std::max(top_right.x(), m_max_pt.x());
        float bbox_max_y = std::max(top_right.y(), m_max_pt.y());
        float bbox_max_z = std::max(top_right.z(), m_max_pt.z());

        m_min_pt = QVector3D(bbox_min_x, bbox_min_y, bbox_min_z);
        m_max_pt = QVector3D(bbox_max_x, bbox_max_y, bbox_max_z);
    }
    QVector3D minPoint() const { return m_min_pt; }
    QVector3D maxPoint() const { return m_max_pt; }

private:

    RCameraController * m_controller;
    QVector3D m_min_pt;
    QVector3D m_max_pt;

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

    RCameraController * cameraController() const { return m_scene->controller(); }

    QPixmap grabPixmap() const { return m_view->screen()->grabWindow(m_view->winId()); }

    void showMark();
    void updateMesh(std::shared_ptr<StaticMesh> const & mesh);
    void updateWorld(std::shared_ptr<WorldFp64> const & world);

    std::shared_ptr<StaticMesh> mesh() const { return m_mesh; }

private:

    Qt3DExtras::Qt3DWindow * m_view = nullptr;
    RScene * m_scene = nullptr;
    QWidget * m_container = nullptr;
    std::shared_ptr<StaticMesh> m_mesh;

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
