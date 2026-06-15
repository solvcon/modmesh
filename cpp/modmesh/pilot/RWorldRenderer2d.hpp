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
