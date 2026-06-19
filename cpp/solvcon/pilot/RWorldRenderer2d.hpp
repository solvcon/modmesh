#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/World.hpp>

#include <QPointF>

class QPainter;

namespace solvcon
{

/**
 * Renders a world's live points, segments, and curves into a QPainter in
 * screen space, mapping math-convention world coordinates through a 2D view
 * transform. paint() draws geometry only; paint_canvas() adds the backdrop
 * and optional grid/axes/origin chrome.
 * m_world is non-owning and may be null.
 */
class RWorldRenderer2d
{
public:
    RWorldRenderer2d(WorldFp64 const * world, ViewTransform2dFp64 const & view)
        : m_world(world)
        , m_view(view)
    {
    }

    /// Paint backdrop, geometry, and optional chrome.
    /// @param painter Target painter in screen space.
    /// @param width Canvas width in pixels.
    /// @param height Canvas height in pixels.
    /// @param full_canvas If true, draw grid, axes, and the origin marker.
    void paint_canvas(QPainter & painter, int width, int height, bool full_canvas) const;

private:
    void paint(QPainter & painter) const;

    // Map math-convention world (x, y) to Qt screen pixels; z is dropped.
    QPointF map(double world_x, double world_y) const;

    WorldFp64 const * m_world;
    ViewTransform2dFp64 const & m_view;
}; /* end class RWorldRenderer2d */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
