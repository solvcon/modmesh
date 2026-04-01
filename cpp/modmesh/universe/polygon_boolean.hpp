#pragma once

/*
 * Copyright (c) 2025, An-Chi Liu <phy.tiger@gmail.com>
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

/**
 * @file Boolean operations on polygons using trapezoidal decomposition.
 *
 * Provides BooleanDecompositionHelper, a helper class that implements the
 * sweep-line algorithm for polygon boolean operations (union, intersection,
 * difference).  Extracted from compute_boolean_with_decomposition() in
 * polygon.hpp to improve readability and avoid lambda closures in tight loops.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace modmesh
{

// Forward declarations -- full definitions live in polygon.hpp
template <typename T>
class PolygonPad;
template <typename T>
class TrapezoidPad;
template <typename T>
class Point3d;
template <typename T>
class Polygon3d;

namespace detail
{

/**
 * Helper class for boolean operations on two polygons via trapezoidal
 * decomposition.
 *
 * A "band" (Y-band) is a horizontal strip between two consecutive critical
 * Y-values [y_lo, y_hi].  The set of critical Y-values comes from:
 *   - the top/bottom edges of every trapezoid produced by decomposing both
 *     input polygons, and
 *   - the Y-coordinates where a trapezoid edge from one polygon crosses a
 *     trapezoid edge from the other polygon.
 *
 * Within a single band no edges from different polygons cross, so the
 * left-to-right ordering of edges is stable.  Each polygon's trapezoids
 * that span the band contribute X-intervals (start at the left edge, end
 * at the right edge).  Sweeping those interval events along X while
 * tracking inside/outside counts for each polygon lets us apply the
 * boolean predicate (union / intersection / difference) and emit result
 * trapezoids.
 *
 * Illustration for two overlapping squares P1=(0,0)-(2,2) and P2=(1,1)-(3,3):
 *
 *  Critical Y-values: {0, 1, 2, 3}
 *  Three y-bands: [0,1], [1,2], [2,3]
 *
 *   y=3 - - - - - +----------------+- - - - - - - - - -
 *                 |          P2    |        band [2,3]
 *   y=2 +----------------+- - - - -|- - - - - - - - - -
 *       |   P1    |//////|         |        band [1,2]
 *   y=1 |- - - - -+------+---------+- - - - - - - - - -
 *       |                |                  band [0,1]
 *   y=0 +----------------+- - - - - - - - - - - - - - -
 *       x=0      x=1    x=2       x=3
 *
 *   Band [1,2]: P1 interval [0,2], P2 interval [1,3]
 *     Events sorted by x position:  P1-start@x=0, P2-start@x=1, P1-end@x=2, P2-end@x=3
 *     Intersection (P1 AND P2): result interval [1,2]
 *     Union        (P1 OR  P2): result interval [0,3]
 *     Difference   (P1-P2)    : result interval [0,1]
 *
 * The algorithm works by:
 * 1. Decomposing both polygons into trapezoids via TrapezoidalDecomposer
 * 2. Collecting all critical Y-values from both trapezoid sets, plus
 *    Y-values where trapezoid edges from different polygons cross
 * 3. For each Y-band, gathering X-intervals from each polygon's trapezoids
 * 4. Applying the boolean predicate on the interval events
 * 5. Emitting result trapezoids as polygons
 *
 * Usage:
 *   BooleanDecompositionHelper<T> helper(pad, polygon_id1, polygon_id2, op);
 *   auto result = helper.compute();
 *
 * @tparam T floating-point type (float or double)
 */
template <typename T>
class BooleanDecompositionHelper
{

public:

    using value_type = T;
    using point_type = Point3d<T>;

    struct Event
    {
        value_type x_lo, x_hi; // X at y_low and y_high
        int source;
        bool is_start; // true = entering the polygon interval, false = leaving
    }; /* end struct Event */

    BooleanDecompositionHelper(
        const std::shared_ptr<PolygonPad<T>> & pad,
        size_t polygon_id1,
        size_t polygon_id2,
        BooleanOperation op)
        : m_pad(pad)
        , m_polygon_id1(polygon_id1)
        , m_polygon_id2(polygon_id2)
        , m_op(op)
        , m_trap_pad(nullptr)
        , m_begin1(0)
        , m_end1(0)
        , m_begin2(0)
        , m_end2(0)
    {
    }

    std::shared_ptr<PolygonPad<T>> compute()
    {
        std::shared_ptr<PolygonPad<T>> result = PolygonPad<T>::construct(m_pad->ndim());

        // Short-circuit: operating a polygon with itself
        if (m_polygon_id1 == m_polygon_id2)
        {
            if (m_op == BooleanOperation::Difference)
            {
                return result; // P - P = empty
            }
            // Union(P, P) = P, Intersection(P, P) = P: copy the polygon
            Polygon3d<T> poly = m_pad->get_polygon(m_polygon_id1);
            std::vector<point_type> nodes;
            nodes.reserve(poly.nnode());
            for (size_t i = 0; i < poly.nnode(); ++i)
            {
                nodes.push_back(poly.node(i));
            }
            result->add_polygon(nodes);
            return result;
        }

        // Step 1: Decompose both polygons using the pad's TrapezoidalDecomposer
        std::tie(m_begin1, m_end1) = m_pad->decompose_to_trapezoid(m_polygon_id1);
        std::tie(m_begin2, m_end2) = m_pad->decompose_to_trapezoid(m_polygon_id2);
        m_trap_pad = m_pad->decomposed_trapezoids();

        if (m_begin1 == m_end1 && m_begin2 == m_end2)
        {
            return result;
        }

        // Step 2: Read trapezoid geometry directly from trap_pad.
        //
        // Each trapezoid in the TrapezoidPad has 4 corners stored as:
        //   p0 = (x0, y0) bottom-left     p1 = (x1, y1) bottom-right
        //   p3 = (x3, y3) top-left        p2 = (x2, y2) top-right
        //
        // A trapezoid forms a horizontal band [y0, y3] (bottom to top).
        // Its left edge goes from x0 (at y0) to x3 (at y3); its right edge
        // goes from x1 (at y0) to x2 (at y3).  Both edges are linear in Y.
        //
        // The source polygon is determined by the index range:
        //   [begin1, end1) -> polygon1 (source 0)
        //   [begin2, end2) -> polygon2 (source 1)

        // Collect critical Y-values
        std::set<value_type> y_set;
        collect_y_values(y_set);
        collect_crossing_y_values(y_set);

        // Convert to sorted vector and merge near-duplicate Y-values that
        // differ only by floating-point rounding, avoiding degenerate
        // near-zero-height bands.
        std::vector<value_type> y_values(y_set.begin(), y_set.end());
        {
            auto merged_end = std::unique(y_values.begin(), y_values.end(), [](value_type a, value_type b)
                                          {
                                              value_type scale = std::max(std::abs(a), std::abs(b));
                                              return std::abs(a - b) < eps * std::max(scale, value_type(1)); });
            y_values.erase(merged_end, y_values.end());
        }

        if (y_values.size() < 2)
        {
            return result;
        }

        // Step 3: Sweep through Y-bands.
        // Each consecutive pair [y_values[yi], y_values[yi+1]] forms one band.
        // Within this band we collect X-interval events from every trapezoid
        // that vertically spans it, then sweep those events left-to-right to
        // determine which X-regions satisfy the boolean predicate.

        // Reuse the events vector across bands to avoid repeated heap allocation
        std::vector<Event> events;
        for (size_t yi = 0; yi + 1 < y_values.size(); ++yi)
        {
            value_type y_low = y_values[yi];
            value_type y_high = y_values[yi + 1];

            events.clear();
            gather_events(events, m_begin1, m_end1, y_low, y_high);
            gather_events(events, m_begin2, m_end2, y_low, y_high);

            if (events.empty())
            {
                continue;
            }

            sort_events(events);
            sweep_events(events, y_low, y_high, result);
        }

        return result;
    }

private:

    static constexpr value_type eps = std::numeric_limits<value_type>::epsilon() * 100;

    /**
     * Determine which source polygon (0 or 1) a trapezoid index belongs to.
     */
    int source_of(size_t idx) const
    {
        if (idx >= m_begin1 && idx < m_end1)
        {
            return 0;
        }
        if (idx >= m_begin2 && idx < m_end2)
        {
            return 1;
        }
        throw std::logic_error("trapezoid index belongs to neither polygon");
    }

    /**
     * Linearly interpolate X along a trapezoid edge at a given Y.
     * An edge is defined by two points: bottom_point and top_point.
     */
    static value_type lerp_x(const point_type & bottom_point, const point_type & top_point, value_type y)
    {
        value_type dy = top_point.y() - bottom_point.y();
        value_type scale = std::max(std::abs(top_point.y()), std::abs(bottom_point.y()));
        if (std::abs(dy) < eps * std::max(scale, value_type(1)))
        {
            return bottom_point.x();
        }
        value_type t = (y - bottom_point.y()) / dy;
        return bottom_point.x() + t * (top_point.x() - bottom_point.x());
    }

    /**
     * Find the Y-value where two edges cross within an overlap range.
     * Each edge is defined by its X-values at the bottom and top of the range.
     * If a crossing exists, it is inserted into y_set.
     */
    void find_crossing(
        value_type edge_a_x_at_bottom,
        value_type edge_a_x_at_top,
        value_type edge_b_x_at_bottom,
        value_type edge_b_x_at_top,
        value_type overlap_y_bottom,
        value_type overlap_y_top,
        std::set<value_type> & y_set) const
    {
        // Parameterize two edges as linear functions of t in [0,1]:
        //   edge_a(t) = edge_a_x_at_bottom + dx_a * t
        //   edge_b(t) = edge_b_x_at_bottom + dx_b * t
        // They cross when (edge_a_x_at_bottom - edge_b_x_at_bottom) + (dx_a - dx_b)*t = 0
        value_type dx_a = edge_a_x_at_top - edge_a_x_at_bottom;
        value_type dx_b = edge_b_x_at_top - edge_b_x_at_bottom;
        value_type denom = dx_a - dx_b;
        value_type scale = std::max({std::abs(dx_a), std::abs(dx_b), value_type(1)});
        if (std::abs(denom) < eps * scale)
        {
            return; // edges are parallel, no crossing
        }
        value_type t = (edge_b_x_at_bottom - edge_a_x_at_bottom) / denom;
        if (t > 0 && t < 1)
        {
            value_type crossing_y = overlap_y_bottom + t * (overlap_y_top - overlap_y_bottom);
            y_set.insert(crossing_y);
        }
    }

    /**
     * Boolean predicate: decide whether a region should be included based on
     * the per-polygon inside counts and the requested operation.
     */
    bool should_include(int count_p1, int count_p2) const
    {
        switch (m_op)
        {
        case BooleanOperation::Union:
            return (count_p1 > 0) || (count_p2 > 0); // either polygon contributes to this region
        case BooleanOperation::Intersection:
            return (count_p1 > 0) && (count_p2 > 0); // both polygons must contribute to this region
        case BooleanOperation::Difference:
            return (count_p1 > 0) && (count_p2 == 0); // only include regions where polygon 1 contributes and polygon 2 does not
        default:
            return false;
        }
    }

    /**
     * Gather interval-boundary events from trapezoids in [begin, end) that
     * vertically span the band [y_low, y_high].
     */
    void gather_events(std::vector<Event> & events, size_t begin, size_t end, value_type y_low, value_type y_high) const
    {
        for (size_t idx = begin; idx < end; ++idx)
        {
            if (!(m_trap_pad->y0(idx) <= y_low && m_trap_pad->y3(idx) >= y_high))
            {
                continue; // trapezoid doesn't vertically span this band
            }
            int source = source_of(idx);
            // Left edge: p0 (bottom-left) -> p3 (top-left)
            // Right edge: p1 (bottom-right) -> p2 (top-right)
            point_type left_bottom = m_trap_pad->p0(idx);
            point_type left_top = m_trap_pad->p3(idx);
            point_type right_bottom = m_trap_pad->p1(idx);
            point_type right_top = m_trap_pad->p2(idx);
            value_type left_x_at_band_bottom = lerp_x(left_bottom, left_top, y_low);
            value_type left_x_at_band_top = lerp_x(left_bottom, left_top, y_high);
            value_type right_x_at_band_bottom = lerp_x(right_bottom, right_top, y_low);
            value_type right_x_at_band_top = lerp_x(right_bottom, right_top, y_high);
            events.push_back({left_x_at_band_bottom, left_x_at_band_top, source, true});
            events.push_back({right_x_at_band_bottom, right_x_at_band_top, source, false});
        }
    }

    /**
     * Sort events by X position (use average of x_lo and x_hi).
     * Use tolerance-based comparison to avoid nondeterministic ordering
     * from floating-point rounding at nearly-identical X positions.
     * Tie-breaking: end events before start events.
     */
    static void sort_events(std::vector<Event> & events)
    {
        std::sort(events.begin(), events.end(), [](const Event & a, const Event & b)
                  {
                      value_type xa = a.x_lo + a.x_hi;
                      value_type xb = b.x_lo + b.x_hi;
                      value_type scale = std::max({std::abs(xa), std::abs(xb), value_type(1)});
                      if (std::abs(xa - xb) > eps * scale)
                      {
                          return xa < xb;
                      }
                      // At same X, end events before start events
                      return !a.is_start && b.is_start; });
    }

    /**
     * Collect all unique Y values from trapezoid boundaries of both polygons.
     *
     * TODO: potential performance issue: std container
     */
    void collect_y_values(std::set<value_type> & y_set) const
    {
        for (size_t idx = m_begin1; idx < m_end1; ++idx)
        {
            y_set.insert(m_trap_pad->y0(idx));
            y_set.insert(m_trap_pad->y3(idx));
        }
        for (size_t idx = m_begin2; idx < m_end2; ++idx)
        {
            y_set.insert(m_trap_pad->y0(idx));
            y_set.insert(m_trap_pad->y3(idx));
        }
    }

    /**
     * Find Y values where trapezoid edges from different polygons cross.
     * This ensures no two cross-polygon edges swap order within a band.
     * Only cross-polygon pairs need checking (polygon1 vs polygon2).
     */
    void collect_crossing_y_values(std::set<value_type> & y_set) const
    {
        for (size_t ti = m_begin1; ti < m_end1; ++ti)
        {
            for (size_t tj = m_begin2; tj < m_end2; ++tj)
            {
                // Y-range overlap between the two trapezoids
                value_type y_lo = std::max(m_trap_pad->y0(ti), m_trap_pad->y0(tj));
                value_type y_hi = std::min(m_trap_pad->y3(ti), m_trap_pad->y3(tj));
                if (y_lo >= y_hi)
                {
                    continue;
                }

                // Trapezoid edges (each edge is a pair of points: bottom -> top):
                //   Left edge:  p0 (bottom-left)  -> p3 (top-left)
                //   Right edge: p1 (bottom-right) -> p2 (top-right)
                point_type trap_i_left_bottom = m_trap_pad->p0(ti);
                point_type trap_i_left_top = m_trap_pad->p3(ti);
                point_type trap_i_right_bottom = m_trap_pad->p1(ti);
                point_type trap_i_right_top = m_trap_pad->p2(ti);
                point_type trap_j_left_bottom = m_trap_pad->p0(tj);
                point_type trap_j_left_top = m_trap_pad->p3(tj);
                point_type trap_j_right_bottom = m_trap_pad->p1(tj);
                point_type trap_j_right_top = m_trap_pad->p2(tj);

                // Interpolate each trapezoid's left/right edges within the Y-overlap range to get X-values at y_lo and y_hi.
                value_type trap_i_left_at_bottom = lerp_x(trap_i_left_bottom, trap_i_left_top, y_lo);
                value_type trap_i_left_at_top = lerp_x(trap_i_left_bottom, trap_i_left_top, y_hi);
                value_type trap_i_right_at_bottom = lerp_x(trap_i_right_bottom, trap_i_right_top, y_lo);
                value_type trap_i_right_at_top = lerp_x(trap_i_right_bottom, trap_i_right_top, y_hi);
                value_type trap_j_left_at_bottom = lerp_x(trap_j_left_bottom, trap_j_left_top, y_lo);
                value_type trap_j_left_at_top = lerp_x(trap_j_left_bottom, trap_j_left_top, y_hi);
                value_type trap_j_right_at_bottom = lerp_x(trap_j_right_bottom, trap_j_right_top, y_lo);
                value_type trap_j_right_at_top = lerp_x(trap_j_right_bottom, trap_j_right_top, y_hi);

                // Check all 4 edge-pair crossings between the two trapezoids:
                // left_i vs left_j, left_i vs right_j, right_i vs left_j, right_i vs right_j
                // Add any crossing Y-values to the set of critical Y-values.
                find_crossing(trap_i_left_at_bottom, trap_i_left_at_top, trap_j_left_at_bottom, trap_j_left_at_top, y_lo, y_hi, y_set);
                find_crossing(trap_i_left_at_bottom, trap_i_left_at_top, trap_j_right_at_bottom, trap_j_right_at_top, y_lo, y_hi, y_set);
                find_crossing(trap_i_right_at_bottom, trap_i_right_at_top, trap_j_left_at_bottom, trap_j_left_at_top, y_lo, y_hi, y_set);
                find_crossing(trap_i_right_at_bottom, trap_i_right_at_top, trap_j_right_at_bottom, trap_j_right_at_top, y_lo, y_hi, y_set);
            }
        }
    }

    /**
     * Sweep events left to right within a Y-band, applying the boolean
     * predicate and emitting result trapezoids.
     */
    void sweep_events(
        const std::vector<Event> & events,
        value_type y_low,
        value_type y_high,
        std::shared_ptr<PolygonPad<T>> & result) const
    {
        int count_p1 = 0;
        int count_p2 = 0;
        bool currently_included = false;
        value_type left_x_low = 0;
        value_type left_x_high = 0;

        for (const auto & ev : events)
        {
            bool was_included = currently_included;

            if (ev.source == 0)
            {
                count_p1 += ev.is_start ? 1 : -1;
            }
            else
            {
                count_p2 += ev.is_start ? 1 : -1;
            }

            currently_included = should_include(count_p1, count_p2);

            if (!was_included && currently_included)
            {
                // Entering an included region
                left_x_low = ev.x_lo;
                left_x_high = ev.x_hi;
            }
            else if (was_included && !currently_included)
            {
                // Leaving an included region -- emit trapezoid as polygon (CCW)
                value_type bottom_width = std::abs(ev.x_lo - left_x_low);
                value_type top_width = std::abs(ev.x_hi - left_x_high);
                // Skip degenerate zero-area trapezoids (edges touching at a point/line)
                if (bottom_width > eps || top_width > eps)
                {
                    std::vector<point_type> nodes = {
                        point_type(left_x_low, y_low, 0),
                        point_type(ev.x_lo, y_low, 0),
                        point_type(ev.x_hi, y_high, 0),
                        point_type(left_x_high, y_high, 0)};
                    result->add_polygon(nodes);
                }
            }
        }
    }

    std::shared_ptr<PolygonPad<T>> m_pad;
    size_t m_polygon_id1;
    size_t m_polygon_id2;
    BooleanOperation m_op;
    std::shared_ptr<TrapezoidPad<T>> m_trap_pad;
    size_t m_begin1;
    size_t m_end1;
    size_t m_begin2;
    size_t m_end2;

}; /* end class BooleanDecompositionHelper */

} /* end namespace detail */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
