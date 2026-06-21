/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/R2DWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RWorldRenderer2d.hpp>

#include <array>
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

// Drags spanning fewer than this many screen pixels commit nothing, so a
// stray click in a shape tool does not drop a degenerate shape into the
// world.
constexpr double MIN_DRAW_DRAG_PX = 2.0;

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
    , m_tool(make_draw_tool(default_draw_tool_name()))
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

void R2DWidget::setDrawTool(std::string const & name)
{
    if (name == drawTool())
    {
        return;
    }
    // make_draw_tool throws for an unknown name; let it propagate before any
    // state changes so an invalid request leaves the current tool untouched.
    std::unique_ptr<DrawToolBase> tool = make_draw_tool(name);
    m_tool = std::move(tool);
    // A tool switch abandons any in-progress drag without committing.
    m_drawing = false;
    // A crosshair signals draw mode; the pan tool keeps the default arrow.
    if (m_tool->can_draw_shape())
    {
        setCursor(Qt::CrossCursor);
    }
    else
    {
        unsetCursor();
    }
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

    // Rubber-band preview of the shape currently being dragged, if any.
    paintDrawPreview(painter);
}

void R2DWidget::paintDrawPreview(QPainter & painter) const
{
    if (!m_drawing)
    {
        return;
    }
    // avoid painting when the canvas is just re-entered
    if (m_draw_current_x == m_draw_start_x && m_draw_current_y == m_draw_start_y)
    {
        return;
    }
    std::array<DrawPoint, 2> const points{{{m_draw_start_x, m_draw_start_y}, {m_draw_current_x, m_draw_current_y}}};
    m_tool->paint_preview(painter, m_view, points);
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
        QPointF const pos = event->position();
        if (m_tool->can_draw_shape())
        {
            // Anchor the drag in world space so it is robust to any pan or
            // zoom that happens mid-stroke.
            m_view.world_from_screen(pos.x(), pos.y(), m_draw_start_x, m_draw_start_y);
            m_draw_current_x = m_draw_start_x;
            m_draw_current_y = m_draw_start_y;
            m_drawing = true;
            event->accept();
            return;
        }
        m_panning = true;
        m_last_mouse_pos = pos;
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void R2DWidget::mouseMoveEvent(QMouseEvent * event)
{
    QPointF const pos = event->position();
    if (m_drawing)
    {
        m_view.world_from_screen(pos.x(), pos.y(), m_draw_current_x, m_draw_current_y);
        update();
        event->accept();
        return;
    }
    if (m_panning)
    {
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
    if (event->button() == Qt::LeftButton && m_drawing)
    {
        m_drawing = false;

        double const diff = std::hypot(m_draw_current_x - m_draw_start_x, m_draw_current_y - m_draw_start_y);
        double const drag_px = m_view.zoom() * diff;

        if (m_world && drag_px >= MIN_DRAW_DRAG_PX)
        {
            std::array<DrawPoint, 2> const points{
                {{m_draw_start_x, m_draw_start_y},
                 {m_draw_current_x, m_draw_current_y}}};
            m_tool->commit(*m_world, points);
        }

        update();
        event->accept();
        return;
    }
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
