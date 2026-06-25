#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <QMatrix4x4>
#include <QVector3D>

#include <string>

namespace solvcon
{

/**
 * @brief A single camera controller with two interaction modes.
 *
 * PanZoom frames a 2D domain head-on: dragging pans and the wheel zooms an
 * orthographic view. FirstPerson flies through a 3D domain: dragging looks
 * around and the wheel dollies along the view direction. There are no
 * per-mode camera classes; the controller holds one pose (position, target,
 * up) plus an orthographic zoom factor, and feeds a QMatrix4x4 view matrix.
 *
 * The widget drives the controller from mouse and key events, and Python
 * drives the same primitives (rotate / zoom / pan) and reads or sets the pose
 * directly, so the domain navigates the same way from code as from the mouse.
 */
class RDomainCameraController
{

public:

    enum class Mode
    {
        PanZoom, ///< 2D: drag pans, wheel zooms the orthographic box.
        FirstPerson, ///< 3D: drag looks around, wheel dollies forward.
    };

    void setMode(Mode mode) { m_mode = mode; }
    Mode mode() const { return m_mode; }

    static Mode modeFromName(std::string const & name);
    static std::string modeName(Mode mode);

    /// Frame the camera onto the bounding box [lo, hi] for an @p ndim domain
    /// at the given viewport @p aspect (width / height).
    void fitToBoundingBox(QVector3D const & lo, QVector3D const & hi, uint32_t ndim, float aspect);

    /// The view matrix for the current pose.
    QMatrix4x4 viewMatrix() const;

    /// The orthographic zoom factor (smaller means zoomed in).
    float orthoScale() const { return m_ortho_scale; }

    /// Drag interaction: pan in PanZoom, look around in FirstPerson.
    /// @p dx and @p dy are pixel deltas.
    void rotate(float dx, float dy);

    /// Pan the view in its own plane by the pixel deltas (both modes).
    void pan(float dx, float dy);

    /// Wheel interaction: change the orthographic zoom in PanZoom, dolly along
    /// the view direction in FirstPerson. @p steps is the wheel notch count.
    void zoom(float steps);

    /// Move along the view direction (forward, +) or sideways (strafe, +right)
    /// by a fraction of the scene size; used by the first-person keys.
    void moveForward(float amount);
    void moveRight(float amount);

    QVector3D position() const { return m_position; }
    void setPosition(QVector3D const & position) { m_position = position; }
    QVector3D target() const { return m_target; }
    void setTarget(QVector3D const & target) { m_target = target; }
    QVector3D up() const { return m_up; }
    void setUp(QVector3D const & up) { m_up = up; }

private:

    QVector3D forward() const;
    QVector3D rightAxis() const;

    Mode m_mode = Mode::PanZoom;

    QVector3D m_position{0.0f, 0.0f, 1.0f};
    QVector3D m_target{0.0f, 0.0f, 0.0f};
    QVector3D m_up{0.0f, 1.0f, 0.0f};

    float m_ortho_scale = 1.0f; ///< PanZoom zoom factor.
    float m_radius = 1.0f; ///< Scene scale, sets pan and dolly speeds.

}; /* end class RDomainCameraController */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
