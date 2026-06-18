#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cmath>
#include <type_traits>

namespace modmesh
{

/**
 * Affine view transform for a strictly 2D canvas.
 *
 * Maps math-convention world coordinates (+Y up) to Qt screen coordinates
 * (+Y down). The mapping is:
 *
 *     screen_x = zoom * world_x + pan_x
 *     screen_y = pan_y - zoom * world_y      // +Y-up flip
 *
 * The class is intentionally Qt-free so the math can be unit-tested in the
 * `test_nopython` gtest target without linking Qt.
 */
template <typename T>
class ViewTransform2d
{

public:

    static_assert(std::is_floating_point_v<T>, "ViewTransform2d requires floating point");

    using value_type = T;

    ViewTransform2d() = default;
    ViewTransform2d(ViewTransform2d const &) = default;
    ViewTransform2d(ViewTransform2d &&) = default;
    ViewTransform2d & operator=(ViewTransform2d const &) = default;
    ViewTransform2d & operator=(ViewTransform2d &&) = default;
    ~ViewTransform2d() = default;

    /// Screen-pixel x-offset added to scaled world coordinates. At
    /// identity zoom this is the screen column where world x == 0 lands.
    value_type pan_x() const { return m_pan_x; }
    value_type & pan_x() { return m_pan_x; }
    void set_pan_x(value_type v) { m_pan_x = v; }

    /// Screen-pixel y-offset. At identity zoom this is the screen row
    /// where world y == 0 lands. Combined with the +Y-up flip,
    /// increasing screen-y runs downward while increasing world-y runs
    /// upward.
    value_type pan_y() const { return m_pan_y; }
    value_type & pan_y() { return m_pan_y; }
    void set_pan_y(value_type v) { m_pan_y = v; }

    /// Scale factor in screen pixels per world unit. Must stay
    /// positive; callers that mutate it directly own the invariant
    /// (the widget enforces it via `setViewTransform`).
    value_type zoom() const { return m_zoom; }
    value_type & zoom() { return m_zoom; }
    void set_zoom(value_type v) { m_zoom = v; }

    /// Map world coordinates to Qt screen coordinates. See the class
    /// docstring for the affine formula (note the +Y-up flip on
    /// `screen_y`).
    void screen_from_world(T world_x, T world_y, T & screen_x, T & screen_y) const;

    /// Inverse of `screen_from_world`. Undefined when `zoom() == 0`.
    void world_from_screen(T screen_x, T screen_y, T & world_x, T & world_y) const;

    /// Translate the view by a screen-pixel delta.
    void pan(T dx_screen, T dy_screen);

    /**
     * Multiply the zoom by `factor`, anchored at screen point
     * (anchor_screen_x, anchor_screen_y) so the world point currently under
     * that screen point stays put. `factor` must be finite and > 0; values
     * > 1 zoom in.
     */
    void zoom_at(T factor, T anchor_screen_x, T anchor_screen_y);

    /**
     * Cursor-anchored zoom that respects [min_zoom, max_zoom] bounds. The
     * effective zoom never leaves the band, and when the zoom is already at
     * a limit a request that would push beyond it is a no-op (no pan
     * drift). Non-finite or non-positive `factor` is ignored.
     */
    void zoom_at_clamped(T factor, T anchor_screen_x, T anchor_screen_y, T min_zoom, T max_zoom);

    void reset();

private:

    T m_pan_x = T(0);
    T m_pan_y = T(0);
    T m_zoom = T(1); // screen pixels per world unit; must stay > 0.

}; /* end class ViewTransform2d */

template <typename T>
void ViewTransform2d<T>::screen_from_world(T world_x, T world_y, T & screen_x, T & screen_y) const
{
    screen_x = m_zoom * world_x + m_pan_x;
    screen_y = m_pan_y - m_zoom * world_y;
}

template <typename T>
void ViewTransform2d<T>::world_from_screen(T screen_x, T screen_y, T & world_x, T & world_y) const
{
    world_x = (screen_x - m_pan_x) / m_zoom;
    world_y = (m_pan_y - screen_y) / m_zoom;
}

template <typename T>
void ViewTransform2d<T>::pan(T dx_screen, T dy_screen)
{
    m_pan_x += dx_screen;
    m_pan_y += dy_screen;
}

template <typename T>
void ViewTransform2d<T>::zoom_at(T factor, T anchor_screen_x, T anchor_screen_y)
{
    if (!std::isfinite(factor) || !(factor > T(0)))
    {
        return;
    }
    if (!std::isfinite(anchor_screen_x) || !std::isfinite(anchor_screen_y))
    {
        return;
    }
    T const desired = m_zoom * factor;
    if (!std::isfinite(desired))
    {
        return;
    }
    T world_x = T(0);
    T world_y = T(0);
    world_from_screen(anchor_screen_x, anchor_screen_y, world_x, world_y);
    m_zoom = desired;
    // After zoom changes, recompute pan so (world_x, world_y) maps back
    // to (anchor_screen_x, anchor_screen_y).
    m_pan_x = anchor_screen_x - m_zoom * world_x;
    m_pan_y = anchor_screen_y + m_zoom * world_y;
}

template <typename T>
void ViewTransform2d<T>::zoom_at_clamped(T factor, T anchor_screen_x, T anchor_screen_y, T min_zoom, T max_zoom)
{
    if (!std::isfinite(factor) || !(factor > T(0)))
    {
        return;
    }
    if (!std::isfinite(min_zoom) || !std::isfinite(max_zoom) || !(min_zoom > T(0)) || !(max_zoom >= min_zoom))
    {
        return;
    }
    T desired = m_zoom * factor;
    if (!std::isfinite(desired))
    {
        return;
    }
    if (desired < min_zoom)
    {
        desired = min_zoom;
    }
    else if (desired > max_zoom)
    {
        desired = max_zoom;
    }
    if (desired == m_zoom)
    {
        return;
    }
    zoom_at(desired / m_zoom, anchor_screen_x, anchor_screen_y);
}

template <typename T>
void ViewTransform2d<T>::reset()
{
    m_pan_x = T(0);
    m_pan_y = T(0);
    m_zoom = T(1);
}

using ViewTransform2dFp64 = ViewTransform2d<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
