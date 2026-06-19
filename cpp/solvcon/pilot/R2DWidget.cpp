/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/R2DWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RWorldRenderer2d.hpp>

#include <cmath>

#include <QColor>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
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

constexpr double BASE_GRID_SPACING_PX = 64.0;
constexpr double MIN_GRID_SPACING_PX = 16.0;
constexpr double MAX_GRID_SPACING_PX = 256.0;

QColor const BACKGROUND(32, 32, 36);
QColor const MINOR_GRID(64, 64, 70);
QColor const AXIS(200, 200, 80);
QColor const ORIGIN(220, 80, 80);

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
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), BACKGROUND);

    double const widget_w = static_cast<double>(width());
    double const widget_h = static_cast<double>(height());

    // Snap grid spacing to a power of ten times {1, 2, 5} for a readable band.
    double const target_world = BASE_GRID_SPACING_PX / m_view.zoom();
    double const exponent = std::floor(std::log10(target_world));
    double const base = std::pow(10.0, exponent);
    double spacing_world = base;
    for (double mult : {1.0, 2.0, 5.0, 10.0})
    {
        double const candidate = base * mult;
        double const candidate_px = m_view.zoom() * candidate;
        if (candidate_px >= MIN_GRID_SPACING_PX && candidate_px <= MAX_GRID_SPACING_PX)
        {
            spacing_world = candidate;
            break;
        }
        spacing_world = candidate;
    }
    double const spacing_px = m_view.zoom() * spacing_world;

    // Draw minor grid lines in screen space directly.
    QPen minor_pen(MINOR_GRID);
    minor_pen.setCosmetic(true);
    minor_pen.setWidth(1);
    painter.setPen(minor_pen);

    double const first_x = std::fmod(m_view.pan_x(), spacing_px);
    for (double sx = first_x; sx < widget_w; sx += spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    for (double sx = first_x - spacing_px; sx > 0.0; sx -= spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    double const first_y = std::fmod(m_view.pan_y(), spacing_px);
    for (double sy = first_y; sy < widget_h; sy += spacing_px)
    {
        painter.drawLine(QPointF(0.0, sy), QPointF(widget_w, sy));
    }
    for (double sy = first_y - spacing_px; sy > 0.0; sy -= spacing_px)
    {
        painter.drawLine(QPointF(0.0, sy), QPointF(widget_w, sy));
    }

    // Draw the world axes through the origin, if visible.
    QPen axis_pen(AXIS);
    axis_pen.setCosmetic(true);
    axis_pen.setWidth(1);
    painter.setPen(axis_pen);
    if (m_view.pan_y() >= 0.0 && m_view.pan_y() <= widget_h)
    {
        painter.drawLine(QPointF(0.0, m_view.pan_y()), QPointF(widget_w, m_view.pan_y()));
    }
    if (m_view.pan_x() >= 0.0 && m_view.pan_x() <= widget_w)
    {
        painter.drawLine(QPointF(m_view.pan_x(), 0.0), QPointF(m_view.pan_x(), widget_h));
    }

    // World geometry on top of the grid, under the origin marker.
    if (m_world)
    {
        RWorldRenderer2d(*m_world, m_view).paint(painter);
    }

    // Origin dot (cosmetic, fixed pixel size regardless of zoom).
    QPen origin_pen(ORIGIN);
    origin_pen.setCosmetic(true);
    origin_pen.setWidth(6);
    origin_pen.setCapStyle(Qt::RoundCap);
    painter.setPen(origin_pen);
    painter.drawPoint(QPointF(m_view.pan_x(), m_view.pan_y()));
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
