/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RWorldRenderer2d.hpp> // Must be the first include.

#include <cmath>

#include <QColor>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

namespace solvcon
{

namespace
{

QColor const BACKGROUND(32, 32, 36);
QColor const MINOR_GRID(64, 64, 70);
QColor const AXIS(200, 200, 80);
QColor const ORIGIN(220, 80, 80);
QColor const GEOMETRY(120, 180, 240);

constexpr double BASE_GRID_SPACING_PX = 64.0;
constexpr double MIN_GRID_SPACING_PX = 16.0;
constexpr double MAX_GRID_SPACING_PX = 256.0;

// Cosmetic (zoom-independent) screen widths for world geometry.
constexpr double GEOMETRY_LINE_WIDTH_PX = 1.5;
constexpr int GEOMETRY_POINT_WIDTH_PX = 5;

void paint_chrome(QPainter & painter, ViewTransform2dFp64 const & view, int width, int height)
{
    double const widget_w = static_cast<double>(width);
    double const widget_h = static_cast<double>(height);

    // Snap grid spacing to a power of ten times {1, 2, 5} for a readable band.
    double const target_world = BASE_GRID_SPACING_PX / view.zoom();
    double const exponent = std::floor(std::log10(target_world));
    double const base = std::pow(10.0, exponent);
    double spacing_world = base;
    for (double mult : {1.0, 2.0, 5.0, 10.0})
    {
        double const candidate = base * mult;
        double const candidate_px = view.zoom() * candidate;
        if (candidate_px >= MIN_GRID_SPACING_PX && candidate_px <= MAX_GRID_SPACING_PX)
        {
            spacing_world = candidate;
            break;
        }
        spacing_world = candidate;
    }
    double const spacing_px = view.zoom() * spacing_world;

    // Draw minor grid lines in screen space directly.
    QPen minor_pen(MINOR_GRID);
    minor_pen.setCosmetic(true);
    minor_pen.setWidth(1);
    painter.setPen(minor_pen);

    double const first_x = std::fmod(view.pan_x(), spacing_px);
    for (double sx = first_x; sx < widget_w; sx += spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    for (double sx = first_x - spacing_px; sx > 0.0; sx -= spacing_px)
    {
        painter.drawLine(QPointF(sx, 0.0), QPointF(sx, widget_h));
    }
    double const first_y = std::fmod(view.pan_y(), spacing_px);
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
    if (view.pan_y() >= 0.0 && view.pan_y() <= widget_h)
    {
        painter.drawLine(QPointF(0.0, view.pan_y()), QPointF(widget_w, view.pan_y()));
    }
    if (view.pan_x() >= 0.0 && view.pan_x() <= widget_w)
    {
        painter.drawLine(QPointF(view.pan_x(), 0.0), QPointF(view.pan_x(), widget_h));
    }
}

} // anonymous namespace

QPointF RWorldRenderer2d::map(double world_x, double world_y) const
{
    double screen_x = 0.0;
    double screen_y = 0.0;
    m_view.screen_from_world(world_x, world_y, screen_x, screen_y);
    return QPointF(screen_x, screen_y);
}

void RWorldRenderer2d::paint(QPainter & painter) const
{
    if (!m_world)
    {
        return;
    }

    // Segments and flattened curves share one cosmetic stroke pen.
    QPen geom_pen(GEOMETRY);
    geom_pen.setCosmetic(true);
    geom_pen.setWidthF(GEOMETRY_LINE_WIDTH_PX);
    painter.setPen(geom_pen);

    // 1D straight segments
    std::shared_ptr<SegmentPadFp64> segments = m_world->collect_live_segments();
    for (size_t i = 0; i < segments->size(); ++i)
    {
        painter.drawLine(map(segments->x0(i), segments->y0(i)),
                         map(segments->x1(i), segments->y1(i)));
    }

    // Cubic Beziers; QPainterPath flattens them adaptively, so no sampling.
    std::shared_ptr<CurvePadFp64> curves = m_world->collect_live_curves();
    if (curves->size() > 0)
    {
        QPainterPath path;
        for (size_t i = 0; i < curves->size(); ++i)
        {
            Bezier3dFp64 const c = curves->get(i);
            path.moveTo(map(c.x0(), c.y0()));
            path.cubicTo(map(c.x1(), c.y1()), map(c.x2(), c.y2()), map(c.x3(), c.y3()));
        }
        painter.setBrush(Qt::NoBrush); // stroke the outline only, never fill
        painter.drawPath(path);
    }

    // 0D standalone points as dots with a fixed pixel size at any zoom.
    std::shared_ptr<PointPadFp64> const & points = m_world->points();
    if (points->size() > 0)
    {
        QPen point_pen(GEOMETRY);
        point_pen.setCosmetic(true);
        point_pen.setWidth(GEOMETRY_POINT_WIDTH_PX);
        point_pen.setCapStyle(Qt::RoundCap);
        painter.setPen(point_pen);
        for (size_t i = 0; i < points->size(); ++i)
        {
            painter.drawPoint(map(points->x(i), points->y(i)));
        }
    }
}

void RWorldRenderer2d::paint_canvas(QPainter & painter, int width, int height, bool full_canvas) const
{
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(0, 0, width, height, BACKGROUND);

    if (full_canvas)
    {
        paint_chrome(painter, m_view, width, height);
    }

    paint(painter);

    if (full_canvas)
    {
        // Origin dot (cosmetic, fixed pixel size regardless of zoom).
        QPen origin_pen(ORIGIN);
        origin_pen.setCosmetic(true);
        origin_pen.setWidth(6);
        origin_pen.setCapStyle(Qt::RoundCap);
        painter.setPen(origin_pen);
        painter.drawPoint(QPointF(m_view.pan_x(), m_view.pan_y()));
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
