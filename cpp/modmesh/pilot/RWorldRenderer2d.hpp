#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/pilot/common_detail.hpp> // Must be the first include.

#include <modmesh/universe/ViewTransform2d.hpp>
#include <modmesh/universe/World.hpp>

#include <QPointF>

class QPainter;

namespace modmesh
{

/**
 * Renders a world's live points, segments, and curves into a QPainter in
 * screen space, mapping math-convention world coordinates through a 2D view
 * transform. The world and view it draws are held as required construction
 * arguments, so an instance can only exist for a renderable pair. This is the
 * shared routine used by both the on-screen R2DWidget and the offscreen image
 * renderer, so neither path can drift from the other. It paints only the
 * geometry; the caller owns the background, grid, axes, and any origin marker.
 */
class RWorldRenderer2d
{
public:
    RWorldRenderer2d(WorldFp64 const & world, ViewTransform2dFp64 const & view)
        : m_world(world)
        , m_view(view)
    {
    }

    void paint(QPainter & painter) const;

private:
    // Map math-convention world (x, y) to Qt screen pixels; z is dropped.
    QPointF map(double world_x, double world_y) const;

    WorldFp64 const & m_world;
    ViewTransform2dFp64 const & m_view;
}; /* end class RWorldRenderer2d */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
