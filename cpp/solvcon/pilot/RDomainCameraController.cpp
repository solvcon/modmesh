/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainCameraController.hpp> // Must be the first include.

#include <QQuaternion>
#include <QtMath>

#include <algorithm>
#include <cmath>

namespace solvcon
{

namespace
{

constexpr float FOV_DEGREES = 45.0f; // Matches RDomainScene's perspective FOV.
constexpr float LOOK_DEGREES_PER_PIXEL = 0.3f;
constexpr float PAN_PIXELS_PER_EXTENT = 300.0f;
constexpr float ZOOM_PER_STEP = 0.12f;
constexpr float ORTHO_SCALE_MIN = 0.02f;
constexpr float ORTHO_SCALE_MAX = 50.0f;

} /* end namespace */

RDomainCameraController::Mode RDomainCameraController::modeFromName(std::string const & name)
{
    return ("fps" == name) ? Mode::FirstPerson : Mode::PanZoom;
}

std::string RDomainCameraController::modeName(Mode mode)
{
    return (Mode::FirstPerson == mode) ? "fps" : "pan";
}

QVector3D RDomainCameraController::forward() const
{
    QVector3D const dir = m_target - m_position;
    float const length = dir.length();
    return (length > 0.0f) ? dir / length : QVector3D(0.0f, 0.0f, -1.0f);
}

QVector3D RDomainCameraController::rightAxis() const
{
    QVector3D const axis = QVector3D::crossProduct(forward(), m_up);
    float const length = axis.length();
    return (length > 0.0f) ? axis / length : QVector3D(1.0f, 0.0f, 0.0f);
}

void RDomainCameraController::fitToBoundingBox(
    QVector3D const & lo, QVector3D const & hi, uint32_t ndim, float aspect)
{
    QVector3D const center = (lo + hi) * 0.5f;
    float radius = (hi - lo).length() * 0.5f;
    if (radius <= 0.0f)
    {
        radius = 1.0f;
    }
    m_radius = radius;
    m_ortho_scale = 1.0f;
    m_up = QVector3D(0.0f, 1.0f, 0.0f);
    m_target = center;

    if (3 == ndim)
    {
        // Pull back so the bounding sphere fits the tighter of the vertical
        // and horizontal fields of view for this viewport.
        float const safe_aspect = (aspect > 0.0f) ? aspect : 1.0f;
        float const half_v = qDegreesToRadians(FOV_DEGREES) * 0.5f;
        float const half_h = std::atan(std::tan(half_v) * safe_aspect);
        float const distance = radius / std::tan(std::min(half_v, half_h)) * 1.1f;
        QVector3D const dir = QVector3D(0.6f, 0.5f, 1.0f).normalized();
        m_position = center + dir * distance;
    }
    else
    {
        // 2D: head-on, ahead along +z. The distance is immaterial to the
        // orthographic projection.
        m_position = center + QVector3D(0.0f, 0.0f, 2.0f * radius);
    }
}

QMatrix4x4 RDomainCameraController::viewMatrix() const
{
    QMatrix4x4 view;
    view.lookAt(m_position, m_target, m_up);
    return view;
}

void RDomainCameraController::rotate(float dx, float dy)
{
    if (Mode::FirstPerson == m_mode)
    {
        // Look around in place: yaw about the up axis, pitch about the right
        // axis, keeping the eye fixed and swinging the target.
        QVector3D const f0 = forward();
        QQuaternion const yaw = QQuaternion::fromAxisAndAngle(m_up, -dx * LOOK_DEGREES_PER_PIXEL);
        QQuaternion const pitch = QQuaternion::fromAxisAndAngle(rightAxis(), -dy * LOOK_DEGREES_PER_PIXEL);
        QVector3D dir = (yaw * pitch).rotatedVector(f0).normalized();
        // Avoid gimbal lock: do not let the view direction reach the up axis
        // (which would degenerate the right axis and flip the view). Drop the
        // pitch if it would push past the pole, but always allow easing away.
        QVector3D const up = m_up.normalized();
        float const limit = std::cos(qDegreesToRadians(1.0f));
        float const aligned = std::abs(QVector3D::dotProduct(dir, up));
        if (aligned > limit && aligned > std::abs(QVector3D::dotProduct(f0, up)))
        {
            dir = yaw.rotatedVector(f0).normalized();
        }
        float const distance = (m_target - m_position).length();
        m_target = m_position + dir * distance;
    }
    else
    {
        pan(dx, dy);
    }
}

void RDomainCameraController::pan(float dx, float dy)
{
    // A drag across the viewport pans roughly the visible extent, moving in
    // the camera's own plane (the screen right and screen up axes). Screen up
    // is the world up head-on, so the 2D pan is unchanged.
    float const extent = 2.0f * m_radius * m_ortho_scale;
    float const sx = -dx / PAN_PIXELS_PER_EXTENT * extent;
    float const sy = dy / PAN_PIXELS_PER_EXTENT * extent;
    QVector3D const screen_up = QVector3D::crossProduct(rightAxis(), forward()).normalized();
    QVector3D const offset = rightAxis() * sx + screen_up * sy;
    m_position += offset;
    m_target += offset;
}

void RDomainCameraController::zoom(float steps)
{
    if (Mode::FirstPerson == m_mode)
    {
        moveForward(steps * 0.1f);
    }
    else
    {
        m_ortho_scale = std::clamp(
            m_ortho_scale * std::exp(-steps * ZOOM_PER_STEP),
            ORTHO_SCALE_MIN,
            ORTHO_SCALE_MAX);
    }
}

void RDomainCameraController::moveForward(float amount)
{
    QVector3D const step = forward() * (amount * m_radius);
    m_position += step;
    m_target += step;
}

void RDomainCameraController::moveRight(float amount)
{
    QVector3D const step = rightAxis() * (amount * m_radius);
    m_position += step;
    m_target += step;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
