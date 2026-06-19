#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/World.hpp>

#include <memory>

#include <QPointF>
#include <QWidget>

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;
class QWheelEvent;

namespace solvcon
{

/**
 * Strictly-2D drawing widget for the pilot. Paints the geometry of the
 * same `World<double>` that R3DWidget renders, but maps it to screen
 * space through a 2D view transform instead of a Qt3D camera.
 */
class R2DWidget
    : public QWidget
{
    Q_OBJECT

public:

    explicit R2DWidget(QWidget * parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    /// Read-only access to the current view state.
    ViewTransform2dFp64 const & viewTransform() const { return m_view; }

    /// Replace the view state. Non-finite inputs are ignored and zoom
    /// is clamped to the widget's internal bounds, so the widget's
    /// invariants survive any caller. Setting a well-formed transform
    /// also disables the deferred auto-centering that runs on the
    /// first positive-size resize.
    void setViewTransform(ViewTransform2dFp64 const & v);

    /// Re-center the view so the world origin sits at the widget center.
    void resetView();

    /// Update the world being painted by this widget.
    void updateWorld(std::shared_ptr<WorldFp64> const & world);

    /// Get the world currently being painted.
    std::shared_ptr<WorldFp64> const & world() const { return m_world; }

    /// Hook for subsequent stages; current implementation triggers a repaint.
    void requestRepaint() { update(); }

protected:

    void paintEvent(QPaintEvent * event) override;
    void wheelEvent(QWheelEvent * event) override;
    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void resizeEvent(QResizeEvent * event) override;

private:

    void centerViewOnOrigin();

    ViewTransform2dFp64 m_view;
    std::shared_ptr<WorldFp64> m_world;
    bool m_panning = false;
    bool m_view_modified = false;
    QPointF m_last_mouse_pos;

}; /* end class R2DWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
