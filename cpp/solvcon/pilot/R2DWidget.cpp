/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/R2DWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RWorldRenderer2d.hpp>

#include <cmath>

#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

namespace solvcon
{

namespace
{

// One wheel revolution (360 degrees) doubles the zoom.
constexpr double ZOOM_STEP_PER_DEGREE = 1.0 / 360.0;

constexpr double MIN_ZOOM = 1.0e-6;
constexpr double MAX_ZOOM = 1.0e6;

double clamp_zoom(double zoom)
{
    if (!std::isfinite(zoom))
    {
        return 1.0;
    }
    if (zoom < MIN_ZOOM)
    {
        return MIN_ZOOM;
    }
    if (zoom > MAX_ZOOM)
    {
        return MAX_ZOOM;
    }
    return zoom;
}

bool is_finite_view(ViewTransform2dFp64 const & v)
{
    return std::isfinite(v.pan_x()) && std::isfinite(v.pan_y()) && std::isfinite(v.zoom());
}

} // unnamed namespace

R2DWidget::R2DWidget(QWidget * parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    // Defer centering to the first resizeEvent, when geometry is real.
}

void R2DWidget::setViewTransform(ViewTransform2dFp64 const & v)
{
    if (!is_finite_view(v))
    {
        return;
    }
    m_view = v;
    m_view.set_zoom(clamp_zoom(m_view.zoom()));
    // An explicit view disables the deferred auto-centering.
    m_view_modified = true;
    update();
}

void R2DWidget::resetView()
{
    m_view.reset();
    centerViewOnOrigin();
    // Re-enable auto-centering on later resizes.
    m_view_modified = false;
    update();
}

void R2DWidget::updateWorld(std::shared_ptr<WorldFp64> const & world)
{
    m_world = world;
    update();
}

void R2DWidget::centerViewOnOrigin()
{
    m_view.set_pan_x(static_cast<double>(width()) * 0.5);
    m_view.set_pan_y(static_cast<double>(height()) * 0.5);
}

void R2DWidget::paintEvent(QPaintEvent * /*event*/)
{
    QPainter painter(this);
    constexpr bool full_canvas = true;
    RWorldRenderer2d(m_world.get(), m_view).paint_canvas(painter, width(), height(), full_canvas);
}

void R2DWidget::wheelEvent(QWheelEvent * event)
{
    QPointF const pos = event->position();
    double const degrees = static_cast<double>(event->angleDelta().y()) / 8.0;
    double const factor = std::exp(degrees * ZOOM_STEP_PER_DEGREE * std::log(2.0));

    if (!std::isfinite(factor) || !(factor > 0.0))
    {
        event->ignore();
        return;
    }
    m_view.zoom_at_clamped(factor, pos.x(), pos.y(), MIN_ZOOM, MAX_ZOOM);
    m_view_modified = true;
    update();
    event->accept();
}

void R2DWidget::mousePressEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton)
    {
        m_panning = true;
        m_last_mouse_pos = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void R2DWidget::mouseMoveEvent(QMouseEvent * event)
{
    if (m_panning)
    {
        QPointF const pos = event->position();
        QPointF const delta = pos - m_last_mouse_pos;
        m_last_mouse_pos = pos;
        m_view.pan(delta.x(), delta.y());
        m_view_modified = true;
        update();
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void R2DWidget::mouseReleaseEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton && m_panning)
    {
        m_panning = false;
        unsetCursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void R2DWidget::resizeEvent(QResizeEvent * event)
{
    QWidget::resizeEvent(event);
    // Auto-center the origin until the view is set explicitly.
    if (!m_view_modified && width() > 0 && height() > 0)
    {
        centerViewOnOrigin();
        update();
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
