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
#include <modmesh/universe/bernstein.hpp>
#include <modmesh/universe/bezier.hpp>

#include <deque>

namespace modmesh
{

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

    template <typename... Args>
    static std::shared_ptr<World<T>> construct(Args &&... args)
    {
        return std::make_shared<World<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    explicit World(ctor_passkey const &)
        : m_points(point_pad_type::construct(/* ndim */ 3))
        , m_segments(segment_pad_type::construct(/* ndim */ 3))
        , m_curves(curve_pad_type::construct(/* ndim */ 3))
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

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "World: (" << msg << ") i " << i << " >= size " << s);
        }
    }

    std::shared_ptr<point_pad_type> m_points;
    std::shared_ptr<segment_pad_type> m_segments;
    std::shared_ptr<curve_pad_type> m_curves;

}; /* end class World */

using WorldFp32 = World<float>;
using WorldFp64 = World<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
