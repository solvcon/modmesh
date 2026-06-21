/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/DrawTool.hpp>

#include <cmath>
#include <stdexcept>

#include <QColor>
#include <QPainter>
#include <QPen>
#include <QPointF>

namespace solvcon
{

void DrawToolBase::paint_preview(
    QPainter & painter,
    ViewTransform2dFp64 const & view,
    std::span<DrawPoint const> points) const
{
    QPen pen(QColor(240, 200, 120));
    pen.setCosmetic(true);
    pen.setWidthF(1.5);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    paint_outline(painter, view, points);
}

namespace
{

/// The default navigation tool: a left-button drag pans and zooms the view instead of drawing.
class PanTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "pan";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return false; }

    void commit(WorldFp64 &, std::span<DrawPoint const>) const override {}

protected:

    void paint_outline(QPainter &, ViewTransform2dFp64 const &, std::span<DrawPoint const>) const override {}

}; /* end class PanTool */

/// Circle defined by two gesture points: the center and a point on the rim.
class CircleTool : public DrawToolBase
{

public:

    static constexpr char const * NAME = "circle";

    std::string name() const override { return NAME; }

    bool can_draw_shape() const override { return true; }

    void commit(WorldFp64 & world, std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("CircleTool::commit: need at least two points");
        }
        DrawPoint const & center = points.front();
        DrawPoint const & rim = points.back();
        world.add_circle(center.x, center.y, std::hypot(rim.x - center.x, rim.y - center.y));
    }

protected:

    void paint_outline(QPainter & painter,
                       ViewTransform2dFp64 const & view,
                       std::span<DrawPoint const> points) const override
    {
        if (points.size() < 2)
        {
            throw std::invalid_argument("CircleTool::paint_outline: need at least two points");
        }
        DrawPoint const & center = points.front();
        DrawPoint const & rim = points.back();
        double const radius = std::hypot(rim.x - center.x, rim.y - center.y);
        if (radius <= 0.0)
        {
            throw std::invalid_argument("CircleTool::paint_outline: radius must be positive");
        }
        double center_x = 0.0;
        double center_y = 0.0;
        view.screen_from_world(center.x, center.y, center_x, center_y);
        double const radius_px = view.zoom() * radius;
        painter.drawEllipse(QPointF(center_x, center_y), radius_px, radius_px);
    }

}; /* end class CircleTool */

/// The tool registry. The first entry is the default tool a fresh canvas
/// starts with. Add a new shape by writing a `DrawToolBase` subclass above
/// and appending one entry here.
struct ToolEntry
{
    char const * name;
    std::unique_ptr<DrawToolBase> (*make)();
};

ToolEntry const TOOL_TABLE[] = {
    {PanTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<PanTool>(); }},
    {CircleTool::NAME, []() -> std::unique_ptr<DrawToolBase>
     { return std::make_unique<CircleTool>(); }},
};

} /* end anonymous namespace */

std::vector<std::string> const & draw_tool_names()
{
    static std::vector<std::string> const names = []
    {
        std::vector<std::string> out;
        for (ToolEntry const & entry : TOOL_TABLE)
        {
            out.emplace_back(entry.name);
        }
        return out;
    }();
    return names;
}

std::string const & default_draw_tool_name()
{
    static_assert(std::size(TOOL_TABLE) > 0, "TOOL_TABLE must have at least one entry");
    return draw_tool_names().front();
}

std::unique_ptr<DrawToolBase> make_draw_tool(std::string const & name)
{
    for (ToolEntry const & entry : TOOL_TABLE)
    {
        if (name == entry.name)
        {
            return entry.make();
        }
    }
    throw std::invalid_argument("make_draw_tool: unknown draw tool '" + name + "'");
}

bool is_draw_tool(std::string const & name)
{
    for (ToolEntry const & entry : TOOL_TABLE)
    {
        if (name == entry.name)
        {
            return true;
        }
    }
    return false;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
