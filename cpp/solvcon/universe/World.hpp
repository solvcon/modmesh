#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/SimpleCollector.hpp>
#include <solvcon/serialization/SerializableItem.hpp>
#include <solvcon/universe/bernstein.hpp>
#include <solvcon/universe/bezier.hpp>
#include <solvcon/universe/rtree.hpp>

#include <cmath>
#include <deque>
#include <numbers>
#include <vector>

namespace solvcon
{

enum class ShapeType : uint8_t
{
    DEAD = 0, ///< deleted / unused slot

    // 0D shapes
    POINT = 1,

    // 1D shapes
    LINE = 2,
    BEZIER = 3, ///< single cubic Bezier curve

    // 2D shapes
    TRIANGLE = 4,
    RECTANGLE = 5,
    SQUARE = 6, ///< specialization of RECTANGLE with equal side lengths
    ELLIPSE = 7,
    CIRCLE = 8, ///< specialization of ELLIPSE with equal radii
}; /* end of enum class ShapeType */

inline std::string shape_type_name(ShapeType st)
{
    switch (st)
    {
    case ShapeType::DEAD: return "DEAD";
    case ShapeType::POINT: return "point";
    case ShapeType::LINE: return "line";
    case ShapeType::BEZIER: return "bezier";
    case ShapeType::TRIANGLE: return "triangle";
    case ShapeType::RECTANGLE: return "rectangle";
    case ShapeType::SQUARE: return "square";
    case ShapeType::ELLIPSE: return "ellipse";
    case ShapeType::CIRCLE: return "circle";
    default: return "unknown";
    }
}

/// Level of detail for World::describe_state. C++ callers pass the enum; the
/// Python binding accepts the equivalent lower-case string.
enum class DescribeLevel : uint8_t
{
    BASIC = 0, ///< only what the 2D image draws
}; /* end enum class DescribeLevel */

inline DescribeLevel describe_level_from_string(std::string const & level)
{
    if (level == "basic")
    {
        return DescribeLevel::BASIC;
    }
    throw std::invalid_argument(
        std::format("World: describe_state level '{}' not supported", level));
}

/// JSON-serializable view of one shape's rendered 2D geometry: its id, type,
/// bounding box, segment endpoints, and curve control points. The z component
/// is not rendered, so coordinates are 2D.
// TODO: The shared JSON tool formats numbers with std::to_string, which keeps
// only 6 decimal places. Coordinates that differ past the 6th decimal then
// serialize identically and small magnitudes round to 0, which can confuse the
// numeric oracles downstream.
class WorldShapeState : public SerializableItem
{

public:

    using coords_type = std::vector<double>;
    using segment_list_type = std::vector<std::vector<double>>;
    using curve_list_type = std::vector<std::vector<std::vector<double>>>;

    WorldShapeState() = default;
    WorldShapeState(WorldShapeState const &) = default;
    WorldShapeState(WorldShapeState &&) = default;
    WorldShapeState & operator=(WorldShapeState const &) = default;
    WorldShapeState & operator=(WorldShapeState &&) = default;
    ~WorldShapeState() override = default;

    int32_t id() const { return m_id; }
    int32_t & id() { return m_id; }

    ShapeType type() const { return m_type; }
    ShapeType & type() { return m_type; }

    coords_type const & bbox() const { return m_bbox; }
    coords_type & bbox() { return m_bbox; }

    segment_list_type const & segments() const { return m_segments; }
    segment_list_type & segments() { return m_segments; }

    curve_list_type const & curves() const { return m_curves; }
    curve_list_type & curves() { return m_curves; }

    // The shape type serializes to its lower-case name (e.g. "triangle") so
    // the rendered JSON stays human-readable; this is an output-only view.
    MM_DECL_SERIALIZABLE(
        register_member("id", m_id);
        register_member("type", shape_type_name(m_type));
        register_member("bbox", m_bbox);
        register_member("segments", m_segments);
        register_member("curves", m_curves);)

private:

    int32_t m_id = 0;
    ShapeType m_type = ShapeType::DEAD;
    coords_type m_bbox; ///< [min_x, min_y, max_x, max_y]
    segment_list_type m_segments; ///< each [x0, y0, x1, y1]
    curve_list_type m_curves; ///< each four [x, y]

}; /* end class WorldShapeState */

/// JSON view of the whole world: shapes plus the bare segments, bare curves,
/// and free points that also render but belong to no shape.
class WorldState : public SerializableItem
{

public:

    using shape_list_type = std::vector<WorldShapeState>;
    using segment_list_type = std::vector<std::vector<double>>;
    using curve_list_type = std::vector<std::vector<std::vector<double>>>;
    using point_list_type = std::vector<std::vector<double>>;

    WorldState() = default;
    WorldState(WorldState const &) = default;
    WorldState(WorldState &&) = default;
    WorldState & operator=(WorldState const &) = default;
    WorldState & operator=(WorldState &&) = default;
    ~WorldState() override = default;

    shape_list_type const & shapes() const { return m_shapes; }
    shape_list_type & shapes() { return m_shapes; }

    segment_list_type const & segments() const { return m_segments; }
    segment_list_type & segments() { return m_segments; }

    curve_list_type const & curves() const { return m_curves; }
    curve_list_type & curves() { return m_curves; }

    point_list_type const & points() const { return m_points; }
    point_list_type & points() { return m_points; }

    MM_DECL_SERIALIZABLE(
        register_member("shapes", m_shapes);
        register_member("segments", m_segments);
        register_member("curves", m_curves);
        register_member("points", m_points);)

private:

    shape_list_type m_shapes;
    segment_list_type m_segments; ///< bare segments
    curve_list_type m_curves; ///< bare curves
    point_list_type m_points; ///< free points, each [x, y]

}; /* end class WorldState */

/**
 * Lightweight record mapping a shape ID to the segment and curve ranges
 * it owns in the world's pads. A shape may own segments only
 * (triangle, line, rectangle), curves only (ellipse, circle), or both.
 */
struct ShapeRecord
{
    ShapeType type;
    size_t segment_offset; ///< first index in SegmentPad
    size_t segment_count; ///< number of segments this shape occupies
    size_t curve_offset; ///< first index in CurvePad
    size_t curve_count; ///< number of cubic Beziers this shape occupies
}; /* end of struct ShapeRecord */

/**
 * Entry stored in the R-tree: shape ID + bounding box.
 */
template <typename T>
struct ShapeEntry
{
    int32_t shape_id;
    BoundBox3d<T> bbox;
    bool operator==(ShapeEntry const & other) const { return shape_id == other.shape_id; }
}; /* end of struct ShapeEntry */

template <typename T>
struct RTreeValueOps<ShapeEntry<T>, BoundBox3d<T>>
{
    static BoundBox3d<T> calc_bound_box(ShapeEntry<T> const & entry) { return entry.bbox; }
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
    using bbox_type = BoundBox3d<T>;
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
    std::shared_ptr<point_pad_type> const & points() const { return m_points; }

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
     * Add a line segment as a shape. One segment in the pad.
     */
    int32_t add_line(T x0, T y0, T x1, T y1);

    /**
     * Add an axis-aligned rectangle by decomposing it into 4 segments.
     * (x_min, y_min) is the lower-left corner; (x_max, y_max) the upper-right.
     */
    int32_t add_rectangle(T x_min, T y_min, T x_max, T y_max);

    /**
     * Specialization of add_rectangle with equal side lengths. Tagged SQUARE.
     */
    int32_t add_square(T x_min, T y_min, T size);

    /**
     * Add an axis-aligned ellipse as 4 cubic Bezier curves, one per quadrant,
     * using the standard k = 4*(sqrt(2) - 1)/3 circle approximation.
     * Ellipses own curves, not segments.
     */
    int32_t add_ellipse(T cx, T cy, T rx, T ry);

    /**
     * Specialization of add_ellipse with equal radii. Tagged CIRCLE.
     */
    int32_t add_circle(T cx, T cy, T r);

    /**
     * Add a cubic Bezier controlled by four points.
     */
    int32_t add_bezier_shape(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3);

    /**
     * Add a cubic Bezier from a bezier_type struct.
     */
    int32_t add_bezier_shape(bezier_type const & bezier);

    /**
     * Translate all segments and curves belonging to a shape by (dx, dy).
     */
    void translate_shape(int32_t shape_id, value_type dx, value_type dy);

    /**
     * Remove a shape from the R-tree and registry.
     * Segments and curves remain in their pads as dead data; use clear() to reclaim.
     */
    void remove_shape(int32_t shape_id);

    ShapeType shape_type_of(int32_t shape_id) const { return find_shape_or_throw(shape_id).type; }

    /// Undo the most recent shape creation.
    void undo();

    /// Redo the most recent undone shape creation.
    void redo();

    /// Whether a shape creation is available to undo.
    bool can_undo() const { return !m_undo_stack.empty(); }

    /// Whether an undone shape creation is available to redo.
    bool can_redo() const { return !m_redo_stack.empty(); }

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
     * Collect all curves except those belonging to DEAD shapes.
     * Includes bare curves (added via add_bezier) and live shape curves.
     */
    std::shared_ptr<curve_pad_type> collect_live_curves() const;

    /**
     * Remove all geometry entities (points, segments, curves, shapes)
     * from the world. Rebuilds pads from scratch to reclaim memory.
     */
    void clear();

    /**
     * Describe the world state as a JSON-serializable object.
     */
    std::string describe_state(DescribeLevel level = DescribeLevel::BASIC) const;

private:

    /// 2D endpoints of segment i, as [x0, y0, x1, y1].
    std::vector<double> segment_coords(size_t i) const
    {
        return {m_segments->x0(i), m_segments->y0(i), m_segments->x1(i), m_segments->y1(i)};
    }

    /// 2D control points of curve i, as four [x, y] pairs.
    std::vector<std::vector<double>> curve_coords(size_t i) const
    {
        return {{m_curves->x0(i), m_curves->y0(i)},
                {m_curves->x1(i), m_curves->y1(i)},
                {m_curves->x2(i), m_curves->y2(i)},
                {m_curves->x3(i), m_curves->y3(i)}};
    }

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(std::format("World: ({}) i {} >= size {}", msg, i, s));
        }
    }

    bbox_type compute_shape_bbox(ShapeRecord const & rec) const;

    /// Register a new shape owning [segment_offset, segment_offset + segment_count)
    /// in the segment pad and [curve_offset, curve_offset + curve_count) in the
    /// curve pad. Pushes into the registry and R-tree. Either range may be empty.
    int32_t register_shape(ShapeType type,
                           size_t segment_offset,
                           size_t segment_count,
                           size_t curve_offset = 0,
                           size_t curve_count = 0);

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

    /// A shape creation recorded for redo: its id and the type to restore.
    struct ShapeRedoRecord
    {
        int32_t shape_id;
        ShapeType type;
    }; /* end of struct ShapeRedoRecord */

    SimpleCollector<int32_t> m_undo_stack; ///< Created shape ids, oldest first; the back is the next to undo.
    std::vector<ShapeRedoRecord> m_redo_stack; ///< Undone shapes awaiting redo

}; /* end class World */

template <typename T>
int32_t World<T>::register_shape(ShapeType type,
                                 size_t segment_offset,
                                 size_t segment_count,
                                 size_t curve_offset,
                                 size_t curve_count)
{
    auto shape_id = static_cast<int32_t>(m_shape_registry.size());
    m_shape_registry.push_back(ShapeRecord{type, segment_offset, segment_count, curve_offset, curve_count}); // NOLINT(modernize-use-designated-initializers)
    ++m_nshape;
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(m_shape_registry[shape_id])});

    // Creating a shape becomes the newest undoable step and invalidates any
    // pending redo, mirroring the usual editor undo/redo semantics.
    m_undo_stack.push_back(shape_id);
    m_redo_stack.clear();

    return shape_id;
}

template <typename T>
int32_t World<T>::add_triangle(T x0, T y0, T x1, T y1, T x2, T y2)
{
    size_t const offset = m_segments->size();
    m_segments->append(point_type(x0, y0, 0), point_type(x1, y1, 0));
    m_segments->append(point_type(x1, y1, 0), point_type(x2, y2, 0));
    m_segments->append(point_type(x2, y2, 0), point_type(x0, y0, 0));
    return register_shape(ShapeType::TRIANGLE, offset, 3);
}

template <typename T>
int32_t World<T>::add_line(T x0, T y0, T x1, T y1)
{
    size_t const offset = m_segments->size();
    m_segments->append(point_type(x0, y0, 0), point_type(x1, y1, 0));
    return register_shape(ShapeType::LINE, offset, 1);
}

template <typename T>
int32_t World<T>::add_rectangle(T x_min, T y_min, T x_max, T y_max)
{
    size_t const offset = m_segments->size();
    point_type const p00(x_min, y_min, 0);
    point_type const p10(x_max, y_min, 0);
    point_type const p11(x_max, y_max, 0);
    point_type const p01(x_min, y_max, 0);
    m_segments->append(p00, p10);
    m_segments->append(p10, p11);
    m_segments->append(p11, p01);
    m_segments->append(p01, p00);
    return register_shape(ShapeType::RECTANGLE, offset, 4);
}

template <typename T>
int32_t World<T>::add_square(T x_min, T y_min, T size)
{
    // Delegate through the rectangle build path and retag so shape_type_of
    // distinguishes squares from rectangles.
    int32_t const sid = add_rectangle(x_min, y_min, x_min + size, y_min + size);
    m_shape_registry[sid].type = ShapeType::SQUARE;
    return sid;
}

template <typename T>
int32_t World<T>::add_ellipse(T cx, T cy, T rx, T ry)
{
    // Standard cubic-Bezier circle approximation: control-point offset along
    // the tangent is k * radius, with k = 4*(sqrt(2) - 1)/3.
    T const k = T(4) * (std::sqrt(T(2)) - T(1)) / T(3);
    T const kx = k * rx;
    T const ky = k * ry;
    size_t const offset = m_curves->size();
    // Quadrant 1: (cx+rx, cy) -> (cx, cy+ry), sweeping counter-clockwise.
    m_curves->append(
        point_type(cx + rx, cy, 0),
        point_type(cx + rx, cy + ky, 0),
        point_type(cx + kx, cy + ry, 0),
        point_type(cx, cy + ry, 0));
    // Quadrant 2: (cx, cy+ry) -> (cx-rx, cy).
    m_curves->append(
        point_type(cx, cy + ry, 0),
        point_type(cx - kx, cy + ry, 0),
        point_type(cx - rx, cy + ky, 0),
        point_type(cx - rx, cy, 0));
    // Quadrant 3: (cx-rx, cy) -> (cx, cy-ry).
    m_curves->append(
        point_type(cx - rx, cy, 0),
        point_type(cx - rx, cy - ky, 0),
        point_type(cx - kx, cy - ry, 0),
        point_type(cx, cy - ry, 0));
    // Quadrant 4: (cx, cy-ry) -> (cx+rx, cy).
    m_curves->append(
        point_type(cx, cy - ry, 0),
        point_type(cx + kx, cy - ry, 0),
        point_type(cx + rx, cy - ky, 0),
        point_type(cx + rx, cy, 0));
    return register_shape(ShapeType::ELLIPSE, /*seg_off*/ 0, /*seg_cnt*/ 0, offset, 4);
}

template <typename T>
int32_t World<T>::add_circle(T cx, T cy, T r)
{
    int32_t const sid = add_ellipse(cx, cy, r, r);
    m_shape_registry[sid].type = ShapeType::CIRCLE;
    return sid;
}

template <typename T>
int32_t World<T>::add_bezier_shape(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
{
    size_t const offset = m_curves->size();
    m_curves->append(p0, p1, p2, p3);
    return register_shape(ShapeType::BEZIER, /*seg_off*/ 0, /*seg_cnt*/ 0, offset, 1);
}

template <typename T>
int32_t World<T>::add_bezier_shape(bezier_type const & bezier)
{
    size_t const offset = m_curves->size();
    m_curves->append(bezier);
    return register_shape(ShapeType::BEZIER, /*seg_off*/ 0, /*seg_cnt*/ 0, offset, 1);
}

template <typename T>
void World<T>::translate_shape(int32_t shape_id, value_type dx, value_type dy)
{
    ShapeRecord const & rec = find_shape_or_throw(shape_id);
    // Remove old entry from R-tree before modifying segments/curves.
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t const idx = rec.segment_offset + i;
        m_segments->x0(idx) += dx;
        m_segments->y0(idx) += dy;
        m_segments->x1(idx) += dx;
        m_segments->y1(idx) += dy;
    }
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        m_curves->x0(idx) += dx;
        m_curves->y0(idx) += dy;
        m_curves->x1(idx) += dx;
        m_curves->y1(idx) += dy;
        m_curves->x2(idx) += dx;
        m_curves->y2(idx) += dy;
        m_curves->x3(idx) += dx;
        m_curves->y3(idx) += dy;
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
void World<T>::undo()
{
    // Skip ids already killed by a direct remove_shape so undo never tries to
    // drop the same shape twice.
    while (!m_undo_stack.empty())
    {
        int32_t const shape_id = m_undo_stack.back();
        m_undo_stack.pop_back();
        ShapeRecord & rec = m_shape_registry[shape_id];
        if (rec.type == ShapeType::DEAD)
        {
            continue;
        }
        m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
        m_redo_stack.push_back(ShapeRedoRecord{shape_id, rec.type});
        rec.type = ShapeType::DEAD;
        --m_nshape;
        return;
    }
}

template <typename T>
void World<T>::redo()
{
    if (m_redo_stack.empty())
    {
        return;
    }
    ShapeRedoRecord const record = m_redo_stack.back();
    m_redo_stack.pop_back();
    ShapeRecord & rec = m_shape_registry[record.shape_id];
    rec.type = record.type;
    ++m_nshape;
    m_rtree->insert(ShapeEntry<T>{record.shape_id, compute_shape_bbox(rec)});
    m_undo_stack.push_back(record.shape_id);
}

template <typename T>
std::vector<int32_t> World<T>::query_visible(T min_x, T min_y, T max_x, T max_y) const
{
    bbox_type const viewport(min_x, min_y, T(0), max_x, max_y, T(0));
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
std::shared_ptr<typename World<T>::curve_pad_type> World<T>::collect_live_curves() const
{
    // Mark curve indices owned by DEAD shapes.
    small_vector<bool> dead(m_curves->size(), false);
    for (auto const & rec : m_shape_registry)
    {
        if (rec.type != ShapeType::DEAD)
        {
            continue;
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            dead[rec.curve_offset + i] = true;
        }
    }
    auto result = curve_pad_type::construct(/* ndim */ 3);
    for (size_t i = 0; i < m_curves->size(); ++i)
    {
        if (!dead[i])
        {
            // includes both bare curves and curves of live shapes
            result->append(m_curves->get(i));
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
    m_undo_stack.clear();
    m_redo_stack.clear();
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
        size_t const idx = rec.segment_offset + i;
        mn_x = std::min({mn_x, m_segments->x0(idx), m_segments->x1(idx)});
        mn_y = std::min({mn_y, m_segments->y0(idx), m_segments->y1(idx)});
        mx_x = std::max({mx_x, m_segments->x0(idx), m_segments->x1(idx)});
        mx_y = std::max({mx_y, m_segments->y0(idx), m_segments->y1(idx)});
    }
    // Bound curves by the convex hull of their cubic control points. For
    // 4-quadrant cubic-Bezier ellipses this is exactly the ellipse bbox.
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        mn_x = std::min({mn_x, m_curves->x0(idx), m_curves->x1(idx), m_curves->x2(idx), m_curves->x3(idx)});
        mn_y = std::min({mn_y, m_curves->y0(idx), m_curves->y1(idx), m_curves->y2(idx), m_curves->y3(idx)});
        mx_x = std::max({mx_x, m_curves->x0(idx), m_curves->x1(idx), m_curves->x2(idx), m_curves->x3(idx)});
        mx_y = std::max({mx_y, m_curves->y0(idx), m_curves->y1(idx), m_curves->y2(idx), m_curves->y3(idx)});
    }
    return bbox_type(mn_x, mn_y, T(0), mx_x, mx_y, T(0));
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

template <typename T>
std::string World<T>::describe_state(DescribeLevel level) const
{
    (void)level; // Only BASIC exists today; richer levels are added later.

    // Find the segments and curves owned by live shapes
    small_vector<bool> segment_owned(m_segments->size(), false);
    small_vector<bool> curve_owned(m_curves->size(), false);

    WorldState state;
    for (size_t sid = 0; sid < m_shape_registry.size(); ++sid)
    {
        ShapeRecord const & rec = m_shape_registry[sid];
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            segment_owned[rec.segment_offset + i] = true;
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            curve_owned[rec.curve_offset + i] = true;
        }
        if (rec.type == ShapeType::DEAD)
        {
            continue;
        }
        bbox_type const bb = compute_shape_bbox(rec);
        WorldShapeState shape;
        shape.id() = static_cast<int32_t>(sid);
        shape.type() = rec.type;
        shape.bbox() = {bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()};
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            shape.segments().push_back(segment_coords(rec.segment_offset + i));
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            shape.curves().push_back(curve_coords(rec.curve_offset + i));
        }
        state.shapes().push_back(std::move(shape));
    }

    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (!segment_owned[i])
        {
            // already exclude DEAD shape segments, but include bare segments
            state.segments().push_back(segment_coords(i));
        }
    }
    for (size_t i = 0; i < m_curves->size(); ++i)
    {
        if (!curve_owned[i])
        {
            state.curves().push_back(curve_coords(i));
        }
    }
    for (size_t i = 0; i < m_points->size(); ++i)
    {
        point_type const p = m_points->get(i);
        state.points().push_back({p.x(), p.y()});
    }

    return state.to_json();
}

using WorldFp32 = World<float>;
using WorldFp64 = World<double>;

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
