#pragma once

/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/base.hpp>
#include <modmesh/buffer/SimpleCollector.hpp>
#include <modmesh/universe/bernstein.hpp>
#include <modmesh/universe/bezier.hpp>
#include <modmesh/universe/rtree.hpp>

#include <deque>
#include <vector>

namespace modmesh
{

enum class ShapeType : uint8_t
{
    DEAD = 0, ///< deleted / unused slot
    TRIANGLE = 1,
}; /* end of enum class ShapeType */

/**
 * Lightweight record mapping a shape ID to its segment range in the pad.
 */
struct ShapeRecord
{
    ShapeType type;
    size_t segment_offset; ///< first index in SegmentPad
    size_t segment_count; ///< number of segments this shape occupies
}; /* end of struct ShapeRecord */

/**
 * Entry stored in the R-tree: shape ID + bounding box.
 */
template <typename T>
struct ShapeEntry
{
    int32_t shape_id;
    BoundBox2d<T> bbox;
    bool operator==(ShapeEntry const & other) const { return shape_id == other.shape_id; }
}; /* end of struct ShapeEntry */

template <typename T>
struct RTreeValueOps<ShapeEntry<T>, BoundBox2d<T>>
{
    static BoundBox2d<T> calc_bound_box(ShapeEntry<T> const & entry) { return entry.bbox; }
}; /* end of struct RTreeValueOps */

/**
 * Manage all geometry entities.
 */
template <typename T>
class World
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<World<T>>
{

private:

    class ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using size_type = typename NumberBase<int32_t, T>::size_type;
    using point_type = Point3d<T>;
    using segment_type = Segment3d<T>;
    using bezier_type = Bezier3d<T>;
    using point_pad_type = PointPad<T>;
    using segment_pad_type = SegmentPad<T>;
    using curve_pad_type = CurvePad<T>;
    using bbox_type = BoundBox2d<T>;
    using rtree_type = RTree<ShapeEntry<T>, bbox_type>;

    template <typename... Args>
    static std::shared_ptr<World<T>> construct(Args &&... args)
    {
        return std::make_shared<World<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    explicit World(ctor_passkey const &)
        : m_points(point_pad_type::construct(/* ndim */ 3))
        , m_segments(segment_pad_type::construct(/* ndim */ 3))
        , m_curves(curve_pad_type::construct(/* ndim */ 3))
        , m_rtree(std::make_unique<rtree_type>())
    {
    }

    World() = delete;
    World(World const &) = delete;
    World(World &&) = delete;
    World & operator=(World const &) = delete;
    World & operator=(World &&) = delete;
    ~World() = default;

    void add_point(point_type const & vertex)
    {
        m_points->append(vertex);
    }
    void add_point(value_type x, value_type y, value_type z)
    {
        add_point(point_type(x, y, z));
    }
    size_t npoint() const { return m_points->size(); }
    point_type point(size_t i) const { return m_points->get(i); }
    point_type point_at(size_t i) const
    {
        check_size(i, m_points->size(), "point");
        return m_points->get(i);
    }
    std::shared_ptr<point_pad_type> const & points() { return m_points; }

    void add_segment(segment_type const & segment)
    {
        m_segments->append(segment);
        m_bare_segment_indices.push_back(m_segments->size() - 1);
    }
    void add_segment(point_type const & p0, point_type const & p1)
    {
        add_segment(segment_type(p0, p1));
    }
    size_t nsegment() const { return m_segments->size(); }
    segment_type segment(size_t i) const { return m_segments->get(i); }
    segment_type segment_at(size_t i) const
    {
        check_size(i, m_segments->size(), "segment");
        return m_segments->get(i);
    }
    std::shared_ptr<segment_pad_type> const & segments() { return m_segments; }

    void add_bezier(bezier_type const & bezier)
    {
        m_curves->append(bezier);
    }
    void add_bezier(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        m_curves->append(p0, p1, p2, p3);
    }
    size_t nbezier() const { return m_curves->size(); }
    bezier_type bezier(size_t i) const { return m_curves->get(i); }
    bezier_type bezier_at(size_t i) const
    {
        check_size(i, m_curves->size(), "bezier");
        return m_curves->get_at(i);
    }
    std::shared_ptr<curve_pad_type> const & curves() { return m_curves; }

    /**
     * Add a triangle by decomposing it into 3 segments in the pad.
     * Returns the shape ID for later reference.
     */
    int32_t add_triangle(T x0, T y0, T x1, T y1, T x2, T y2);

    /**
     * Translate all segments belonging to a shape by (dx, dy).
     */
    void translate_shape(int32_t shape_id, value_type dx, value_type dy);

    /**
     * Remove a shape from the R-tree and registry.
     * Segments remain in the pad as dead data; use clear() to reclaim.
     */
    void remove_shape(int32_t shape_id);

    ShapeType shape_type_of(int32_t shape_id) const { return find_shape_or_throw(shape_id).type; }

    size_t nshape() const { return m_nshape; }

    /**
     * Query the R-tree for shapes whose bounding box overlaps the viewport.
     */
    std::vector<int32_t> query_visible(T min_x, T min_y, T max_x, T max_y) const;

    /**
     * Collect all segments except those belonging to DEAD shapes.
     * Includes bare segments (added via add_segment) and live shape segments.
     */
    std::shared_ptr<segment_pad_type> collect_live_segments() const;

    /**
     * Remove all geometry entities (points, segments, curves, shapes)
     * from the world. Rebuilds pads from scratch to reclaim memory.
     */
    void clear();

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(std::format("World: ({}) i {} >= size {}", msg, i, s));
        }
    }

    bbox_type compute_shape_bbox(ShapeRecord const & rec) const;

    /// Check if shape_id is valid and not DEAD.
    /// @throw std::out_of_range if shape_id is out of bounds or shape is DEAD.
    void check_shape_id(int32_t shape_id) const;

    ShapeRecord const & find_shape_or_throw(int32_t shape_id) const
    {
        check_shape_id(shape_id);
        return m_shape_registry[shape_id];
    }

    ShapeRecord & find_shape_or_throw(int32_t shape_id)
    {
        check_shape_id(shape_id);
        return m_shape_registry[shape_id];
    }

    std::shared_ptr<point_pad_type> m_points;

    std::shared_ptr<segment_pad_type> m_segments;
    SimpleCollector<size_t> m_bare_segment_indices; ///< indices of segments not owned by any shape

    std::shared_ptr<curve_pad_type> m_curves;

    // TODO: Replace std::vector with a custom SoA container and BoundBoxPad
    // auxiliary class. Consider moving the registry into the R-tree.
    std::vector<ShapeRecord> m_shape_registry;

    size_t m_nshape = 0; ///< count of live (non-DEAD) shapes
    std::unique_ptr<rtree_type> m_rtree; ///< spatial index for shapes for viewport query

}; /* end class World */

template <typename T>
int32_t World<T>::add_triangle(T x0, T y0, T x1, T y1, T x2, T y2)
{
    size_t offset = m_segments->size();
    m_segments->append(point_type(x0, y0, 0), point_type(x1, y1, 0));
    m_segments->append(point_type(x1, y1, 0), point_type(x2, y2, 0));
    m_segments->append(point_type(x2, y2, 0), point_type(x0, y0, 0));

    int32_t shape_id = static_cast<int32_t>(m_shape_registry.size());
    m_shape_registry.push_back(ShapeRecord{ShapeType::TRIANGLE, offset, 3});
    ++m_nshape;
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(m_shape_registry[shape_id])});
    return shape_id;
}

template <typename T>
void World<T>::translate_shape(int32_t shape_id, value_type dx, value_type dy)
{
    ShapeRecord const & rec = find_shape_or_throw(shape_id);
    // Remove old entry from R-tree before modifying segments.
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t idx = rec.segment_offset + i;
        m_segments->x0(idx) += dx;
        m_segments->y0(idx) += dy;
        m_segments->x1(idx) += dx;
        m_segments->y1(idx) += dy;
    }
    // Reinsert with updated bounding box.
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
}

template <typename T>
void World<T>::remove_shape(int32_t shape_id)
{
    ShapeRecord & rec = find_shape_or_throw(shape_id);
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
    rec.type = ShapeType::DEAD;
    --m_nshape;
}

template <typename T>
std::vector<int32_t> World<T>::query_visible(T min_x, T min_y, T max_x, T max_y) const
{
    bbox_type viewport(min_x, min_y, max_x, max_y);
    std::vector<ShapeEntry<T>> hits;
    m_rtree->search(viewport, hits);
    std::vector<int32_t> ids;
    ids.reserve(hits.size());
    for (auto const & entry : hits)
    {
        ids.push_back(entry.shape_id);
    }
    return ids;
}

template <typename T>
std::shared_ptr<typename World<T>::segment_pad_type> World<T>::collect_live_segments() const
{
    // Mark segment indices owned by DEAD shapes.
    small_vector<bool> dead(m_segments->size(), false);
    for (auto const & rec : m_shape_registry)
    {
        if (rec.type != ShapeType::DEAD)
        {
            continue;
        }
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            dead[rec.segment_offset + i] = true;
        }
    }
    auto result = segment_pad_type::construct(/* ndim */ 3);
    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (!dead[i])
        {
            // includes both bare segments and segments of live shapes
            result->append(m_segments->get(i));
        }
    }
    return result;
}

template <typename T>
void World<T>::clear()
{
    m_points = point_pad_type::construct(/* ndim */ 3);
    m_segments = segment_pad_type::construct(/* ndim */ 3);
    m_curves = curve_pad_type::construct(/* ndim */ 3);
    m_bare_segment_indices.clear();
    m_shape_registry.clear();
    m_nshape = 0;
    m_rtree = std::make_unique<rtree_type>();
}

template <typename T>
typename World<T>::bbox_type World<T>::compute_shape_bbox(ShapeRecord const & rec) const
{
    T mn_x = std::numeric_limits<T>::max();
    T mn_y = std::numeric_limits<T>::max();
    T mx_x = std::numeric_limits<T>::lowest();
    T mx_y = std::numeric_limits<T>::lowest();
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t idx = rec.segment_offset + i;
        mn_x = std::min({mn_x, m_segments->x0(idx), m_segments->x1(idx)});
        mn_y = std::min({mn_y, m_segments->y0(idx), m_segments->y1(idx)});
        mx_x = std::max({mx_x, m_segments->x0(idx), m_segments->x1(idx)});
        mx_y = std::max({mx_y, m_segments->y0(idx), m_segments->y1(idx)});
    }
    return bbox_type(mn_x, mn_y, mx_x, mx_y);
}

template <typename T>
void World<T>::check_shape_id(int32_t shape_id) const
{
    if (shape_id < 0 ||
        static_cast<size_t>(shape_id) >= m_shape_registry.size())
    {
        throw std::out_of_range(
            std::format("World: shape_id {} not found", shape_id));
    }
    if (m_shape_registry[shape_id].type == ShapeType::DEAD)
    {
        throw std::invalid_argument(
            std::format("World: shape_id {} is DEAD", shape_id));
    }
}

using WorldFp32 = World<float>;
using WorldFp64 = World<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
