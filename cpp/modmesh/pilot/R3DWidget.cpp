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

#include <modmesh/pilot/R3DWidget.hpp> // Must be the first include.
#include <modmesh/pilot/RAxisMark.hpp>
#include <modmesh/pilot/RStaticMesh.hpp>

namespace modmesh
{

R3DWidget::R3DWidget(Qt3DExtras::Qt3DWindow * window, RScene * scene, QWidget * parent, Qt::WindowFlags f)
    : QWidget(parent, f)
    , m_view(nullptr == window ? new Qt3DExtras::Qt3DWindow : window)
    , m_scene(nullptr == scene ? new RScene : scene)
    , m_container(createWindowContainer(m_view, this, Qt::Widget))
{
    m_view->setRootEntity(m_scene);

    cameraController()->setCamera(m_view->camera());
    cameraController()->reset();

    if (Toggle::instance().fixed().get_show_axis())
    {
        showMark();
    }
}

void R3DWidget::showMark()
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RAxisMark))
        {
            child->deleteLater();
        }
    }
    new RAxisMark(m_scene);
}

void R3DWidget::updateMesh(std::shared_ptr<StaticMesh> const & mesh)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RStaticMesh))
        {
            child->deleteLater();
        }
    }
    new RStaticMesh(mesh, m_scene);
    m_mesh = mesh;
}

void R3DWidget::updateWorld(std::shared_ptr<WorldFp64> const & world)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if ((typeid(*child) == typeid(RLines)) || (typeid(*child) == typeid(RVertices)))
        {
            child->deleteLater();
        }
    }
    new RVertices(world, m_scene);
    new RLines(world, m_scene);

    QVector3D box_min_pt = m_scene->minPoint(); // get the bottom-left corner of bounding box
    QVector3D box_max_pt = m_scene->maxPoint(); // get the top-right corner of bounding box
    QVector3D box_center = (box_min_pt + box_max_pt) * 0.5f; // center point of the bounding box

    /*
     * Calculate the camera distance to fully view the bounding box based on
     * the vertical field of view (FOV) and the vertical extent of the box.
     *
     * Using the relation: tan(fov / 2) = (half of vertical size) / (distance to center),
     * we can rearrange and solve for the distance:
     *
     *     (distance to center) = (half of vertical size) / tan(fov / 2)
     *
     * This gives the minimum distance from the box center along the view direction within
     * the specified FOV. After that, set the far plane properly to ensurce enclose whole
     * bounding box.
     */
    float fov = 45.0f;
    float half_height = (box_max_pt - box_min_pt).length() * 0.5f;
    float dist = half_height / std::tan(qDegreesToRadians(fov) / 2.0f);

    cameraController()->setPosition(box_center + QVector3D(0, 0, dist));
    cameraController()->setViewCenter(box_center);
    cameraController()->setFarPlane(dist + half_height * 2);
}

void R3DWidget::closeAndDestroy()
{
    this->close();
    this->deleteLater();
}

void R3DWidget::resizeEvent(QResizeEvent * event)
{
    QWidget::resizeEvent(event);
    m_view->resize(event->size());
    m_container->resize(event->size());
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
