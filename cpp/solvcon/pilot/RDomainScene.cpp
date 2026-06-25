/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainScene.hpp> // Must be the first include.

#include <QtMath>

#include <algorithm>
#include <cmath>

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

void RDomainScene::fitCameraToScene()
{
    if (!m_has_bbox)
    {
        m_eye = QVector3D(0.0f, 0.0f, 1.0f);
        m_center = QVector3D(0.0f, 0.0f, 0.0f);
        m_up = QVector3D(0.0f, 1.0f, 0.0f);
        return;
    }

    QVector3D const center = (m_bbox_lo + m_bbox_hi) * 0.5f;
    float const radius = boundingRadius();

    // 2D domains are viewed head-on; 3D domains from a fixed oblique angle so
    // depth reads until the interactive camera lands.
    QVector3D const dir = (3 == m_ndim)
                              ? QVector3D(0.6f, 0.5f, 1.0f).normalized()
                              : QVector3D(0.0f, 0.0f, 1.0f);

    float distance = 2.0f * radius;
    if (3 == m_ndim)
    {
        // Pull back far enough that the bounding sphere fills the vertical
        // field of view, with a small margin.
        float const half_fov = qDegreesToRadians(FOV_DEGREES) * 0.5f;
        distance = radius / std::tan(half_fov) * 1.1f;
    }

    m_center = center;
    m_eye = center + dir * distance;
    m_up = QVector3D(0.0f, 1.0f, 0.0f);
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

    QMatrix4x4 view;
    QMatrix4x4 proj;
    if (3 == m_ndim)
    {
        // Keep the framing direction from the fit, but settle the pullback
        // here where the aspect is known: the limiting half-angle is the
        // smaller of the vertical and horizontal field of view, so a portrait
        // viewport pulls the camera further back instead of clipping the
        // sides.
        QVector3D direction = m_eye - m_center;
        float const length = direction.length();
        direction = (length > 0.0f) ? direction / length : QVector3D(0.0f, 0.0f, 1.0f);
        float const half_v = qDegreesToRadians(FOV_DEGREES) * 0.5f;
        float const half_h = std::atan(std::tan(half_v) * aspect);
        float const distance = radius / std::tan(std::min(half_v, half_h)) * 1.1f;
        view.lookAt(m_center + direction * distance, m_center, m_up);
        proj.perspective(FOV_DEGREES, aspect, 0.01f * radius, distance + 2.0f * radius);
    }
    else
    {
        // A bounding sphere of this radius fits from any direction, so the
        // orthographic box is simply sized around it for the viewport aspect.
        view.lookAt(m_eye, m_center, m_up);
        float const margin = radius * 1.1f;
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
        proj.ortho(-half_w, half_w, -half_h, half_h, 0.01f * radius, 5.0f * radius);
    }

    return clip * proj * view;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
