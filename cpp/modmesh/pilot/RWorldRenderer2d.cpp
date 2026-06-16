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

#include <modmesh/pilot/RWorldRenderer2d.hpp> // Must be the first include.

#include <QColor>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

namespace modmesh
{

namespace
{

QColor const GEOMETRY(120, 180, 240);

// Cosmetic (zoom-independent) screen widths for world geometry.
constexpr double GEOMETRY_LINE_WIDTH_PX = 1.5;
constexpr int GEOMETRY_POINT_WIDTH_PX = 5;

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
    // Segments and flattened curves share one cosmetic stroke pen.
    QPen geom_pen(GEOMETRY);
    geom_pen.setCosmetic(true);
    geom_pen.setWidthF(GEOMETRY_LINE_WIDTH_PX);
    painter.setPen(geom_pen);

    // 1D straight segments
    std::shared_ptr<SegmentPadFp64> segments = m_world.collect_live_segments();
    for (size_t i = 0; i < segments->size(); ++i)
    {
        painter.drawLine(map(segments->x0(i), segments->y0(i)),
                         map(segments->x1(i), segments->y1(i)));
    }

    // Cubic Beziers; QPainterPath flattens them adaptively, so no sampling.
    std::shared_ptr<CurvePadFp64> curves = m_world.collect_live_curves();
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
    std::shared_ptr<PointPadFp64> const & points = m_world.points();
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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
