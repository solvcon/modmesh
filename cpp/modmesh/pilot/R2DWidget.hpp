#pragma once

/*
 * Copyright (c) 2026, An-Chi Liu <phy.tiger@gmail.com>
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

#include <modmesh/universe/ViewTransform2d.hpp>
#include <modmesh/universe/World.hpp>

#include <memory>

#include <QPointF>
#include <QWidget>

class QMouseEvent;
class QPainter;
class QPaintEvent;
class QResizeEvent;
class QWheelEvent;

namespace modmesh
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

    /// Paint the world's live points, segments, and curves in screen space.
    void paintWorld(QPainter & painter) const;

    ViewTransform2dFp64 m_view;
    std::shared_ptr<WorldFp64> m_world;
    bool m_panning = false;
    bool m_view_modified = false;
    QPointF m_last_mouse_pos;

}; /* end class R2DWidget */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
