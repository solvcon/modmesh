/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainScene.hpp> // Must be the first include.

#include <algorithm>

namespace solvcon
{

namespace
{

// Vertical field of view used for the 3D perspective projection.
constexpr float FOV_DEGREES = 45.0f;

} /* end namespace */

void RDomainScene::addDrawable(std::unique_ptr<RDrawable> drawable)
{
    m_drawables.push_back(std::move(drawable));
}

void RDomainScene::removeDrawable(RDrawable const * drawable)
{
    if (nullptr == drawable)
    {
        return;
    }
    std::erase_if(
        m_drawables,
        [drawable](std::unique_ptr<RDrawable> const & d)
        { return d.get() == drawable; });
}

void RDomainScene::removeDrawableIf(std::function<bool(RDrawable const *)> const & pred)
{
    std::erase_if(
        m_drawables,
        [&pred](std::unique_ptr<RDrawable> const & d)
        { return pred(d.get()); });
}

void RDomainScene::releaseAll()
{
    for (std::unique_ptr<RDrawable> const & drawable : m_drawables)
    {
        drawable->release();
    }
}

void RDomainScene::resetBoundingBox()
{
    m_has_bbox = false;
}

void RDomainScene::extendBoundingBox(QVector3D const & lo, QVector3D const & hi)
{
    if (!m_has_bbox)
    {
        m_bbox_lo = lo;
        m_bbox_hi = hi;
        m_has_bbox = true;
        return;
    }
    m_bbox_lo = QVector3D(
        std::min(m_bbox_lo.x(), lo.x()),
        std::min(m_bbox_lo.y(), lo.y()),
        std::min(m_bbox_lo.z(), lo.z()));
    m_bbox_hi = QVector3D(
        std::max(m_bbox_hi.x(), hi.x()),
        std::max(m_bbox_hi.y(), hi.y()),
        std::max(m_bbox_hi.z(), hi.z()));
}

float RDomainScene::boundingRadius() const
{
    float const radius = (m_bbox_hi - m_bbox_lo).length() * 0.5f;
    return (radius > 0.0f) ? radius : 1.0f;
}

void RDomainScene::fitCameraToScene(float aspect)
{
    if (!m_has_bbox)
    {
        m_camera.setPosition(QVector3D(0.0f, 0.0f, 1.0f));
        m_camera.setTarget(QVector3D(0.0f, 0.0f, 0.0f));
        m_camera.setUp(QVector3D(0.0f, 1.0f, 0.0f));
        return;
    }
    m_camera.fitToBoundingBox(m_bbox_lo, m_bbox_hi, m_ndim, aspect);
}

QMatrix4x4 RDomainScene::viewProjection(QSize pixel_size, QRhi * rhi) const
{
    QMatrix4x4 clip = (nullptr != rhi) ? rhi->clipSpaceCorrMatrix() : QMatrix4x4();
    if (!m_has_bbox || pixel_size.height() <= 0 || pixel_size.width() <= 0)
    {
        return clip;
    }

    float const aspect = static_cast<float>(pixel_size.width()) / static_cast<float>(pixel_size.height());
    float const radius = boundingRadius();

    QMatrix4x4 const view = m_camera.viewMatrix();
    float distance = (m_camera.position() - m_camera.target()).length();
    if (distance <= 0.0f)
    {
        distance = 2.0f * radius;
    }

    QMatrix4x4 proj;
    if (3 == m_ndim)
    {
        proj.perspective(FOV_DEGREES, aspect, 0.01f * radius, distance + 3.0f * radius);
    }
    else
    {
        // The orthographic box is sized around the bounding sphere for the
        // viewport aspect, scaled by the camera's zoom factor.
        float const margin = radius * 1.1f * m_camera.orthoScale();
        float half_w = margin;
        float half_h = margin;
        if (aspect >= 1.0f)
        {
            half_w = margin * aspect;
        }
        else
        {
            half_h = margin / aspect;
        }
        proj.ortho(-half_w, half_w, -half_h, half_h, 0.01f * radius, distance + 3.0f * radius);
    }

    return clip * proj * view;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
