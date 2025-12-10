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

#include <modmesh/universe/polygon.hpp>

namespace modmesh
{

template <typename T>
SimpleArray<T> TrianglePad<T>::pack_array() const
{
    using shape_type = typename SimpleArray<T>::shape_type;
    SimpleArray<T> ret(shape_type{m_p0->size(), static_cast<size_t>(ndim() * 3)});
    if (ndim() == 3)
    {
        for (size_t i = 0; i < m_p0->size(); ++i)
        {
            ret(i, 0) = m_p0->x(i);
            ret(i, 1) = m_p0->y(i);
            ret(i, 2) = m_p0->z(i);
            ret(i, 3) = m_p1->x(i);
            ret(i, 4) = m_p1->y(i);
            ret(i, 5) = m_p1->z(i);
            ret(i, 6) = m_p2->x(i);
            ret(i, 7) = m_p2->y(i);
            ret(i, 8) = m_p2->z(i);
        }
    }
    else
    {
        for (size_t i = 0; i < m_p0->size(); ++i)
        {
            ret(i, 0) = m_p0->x(i);
            ret(i, 1) = m_p0->y(i);
            ret(i, 2) = m_p1->x(i);
            ret(i, 3) = m_p1->y(i);
            ret(i, 4) = m_p2->x(i);
            ret(i, 5) = m_p2->y(i);
        }
    }
    return ret;
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon(std::vector<point_type> const & nodes)
{
    if (nodes.empty())
    {
        throw std::invalid_argument("PolygonPad::add_polygon: cannot add empty polygon");
    }

    ssize_type const begin_index = static_cast<ssize_type>(m_points->size());

    for (point_type const & node : nodes)
    {
        m_points->append(node);
    }

    ssize_type const end_index = static_cast<ssize_type>(m_points->size());
    size_t const polygon_id = m_begins.size();
    m_begins.push_back(begin_index);
    m_ends.push_back(end_index);

    polygon_type polygon(this->shared_from_this(), polygon_id, typename polygon_type::ctor_passkey());
    rebuild_polygon_rtree(polygon);

    return polygon;
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_segments(std::shared_ptr<segment_pad_type> segments)
{
    if (segments->size() == 0)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_segments: empty segment pad");
    }

    std::vector<point_type> nodes;
    nodes.reserve(segments->size());

    for (size_t i = 0; i < segments->size(); ++i)
    {
        nodes.push_back(segments->p0(i));
    }

    return add_polygon(nodes);
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_curves(std::shared_ptr<curve_pad_type> curves, value_type sample_length)
{
    std::shared_ptr<segment_pad_type> segments = curves->sample(sample_length);
    return add_polygon_from_segments(segments);
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_segments_and_curves(
    std::shared_ptr<segment_pad_type> segments,
    std::shared_ptr<curve_pad_type> curves,
    value_type sample_length)
{
    std::vector<point_type> nodes;

    for (size_t i = 0; i < segments->size(); ++i)
    {
        nodes.push_back(segments->p0(i));
    }

    std::shared_ptr<segment_pad_type> curve_segments = curves->sample(sample_length);
    for (size_t i = 0; i < curve_segments->size(); ++i)
    {
        nodes.push_back(curve_segments->p0(i));
    }

    return add_polygon(nodes);
}

template <typename T>
size_t PolygonPad<T>::get_num_nodes(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_num_nodes: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    return static_cast<size_t>(end_index - begin_index);
}

template <typename T>
Point3d<T> PolygonPad<T>::get_node(size_t polygon_id, size_t node_index) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_node: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    size_t const count = static_cast<size_t>(end_index - begin_index);
    if (node_index >= count)
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_node: node_index {} >= count {}",
                        node_index,
                        count));
    }
    return m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(node_index)));
}

template <typename T>
Segment3d<T> PolygonPad<T>::get_edge(size_t polygon_id, size_t edge_index) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_edge: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    size_t const count = static_cast<size_t>(end_index - begin_index);
    if (edge_index >= count)
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_edge: edge_index {} >= count {}",
                        edge_index,
                        count));
    }
    point_type const p0 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(edge_index)));
    point_type const p1 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>((edge_index + 1) % count)));
    return segment_type(p0, p1);
}

template <typename T>
T PolygonPad<T>::compute_signed_area(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::compute_signed_area: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    auto const begin_index = m_begins[polygon_id];
    auto const end_index = m_ends[polygon_id];
    size_t const count = static_cast<size_t>(end_index - begin_index);
    if (count < 3)
    {
        return 0;
    }

    value_type area = 0;
    for (size_t i = 0; i < count; ++i)
    {
        point_type const p0 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(i)));
        point_type const p1 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>((i + 1) % count)));
        area += p0.x() * p1.y() - p1.x() * p0.y();
    }

    return area / 2;
}

template <typename T>
BoundBox3d<T> PolygonPad<T>::calc_bound_box(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::calc_bound_box: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    auto const begin_index = m_begins[polygon_id];
    auto const end_index = m_ends[polygon_id];
    size_t const count = static_cast<size_t>(end_index - begin_index);

    if (count == 0)
    {
        return BoundBox3d<T>(0, 0, 0, 0, 0, 0);
    }

    value_type min_x = std::numeric_limits<value_type>::max();
    value_type min_y = std::numeric_limits<value_type>::max();
    value_type min_z = std::numeric_limits<value_type>::max();
    value_type max_x = std::numeric_limits<value_type>::lowest();
    value_type max_y = std::numeric_limits<value_type>::lowest();
    value_type max_z = std::numeric_limits<value_type>::lowest();

    for (size_t i = 0; i < count; ++i)
    {
        point_type const node = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(i)));
        min_x = std::min(min_x, node.x());
        min_y = std::min(min_y, node.y());
        min_z = std::min(min_z, node.z());
        max_x = std::max(max_x, node.x());
        max_y = std::max(max_y, node.y());
        max_z = std::max(max_z, node.z());
    }

    return BoundBox3d<T>(min_x, min_y, min_z, max_x, max_y, max_z);
}

template <typename T>
void PolygonPad<T>::rebuild_rtree()
{
    m_rtree = std::make_unique<rtree_type>();
    for (size_t i = 0; i < m_begins.size(); ++i)
    {
        polygon_type polygon = get_polygon(i);
        rebuild_polygon_rtree(polygon);
    }
}

template <typename T>
void PolygonPad<T>::rebuild_polygon_rtree(polygon_type const & polygon)
{
    size_t const count = polygon.nnode();

    for (size_t i = 0; i < count; ++i)
    {
        segment_type const edge = polygon.edge(i);
        m_rtree->insert(edge);
    }
}

template class TrianglePad<float>;
template class TrianglePad<double>;

template class PolygonPad<float>;
template class PolygonPad<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
