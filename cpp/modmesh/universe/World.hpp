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
    using vector_type = Vector3d<T>;
    using vertex_type = vector_type;
    using edge_type = Edge3d<T>;
    using bezier_type = Bezier3d<T>;

    template <typename... Args>
    static std::shared_ptr<World<T>> construct(Args &&... args)
    {
        return std::make_shared<World<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    explicit World(ctor_passkey const &) {}

    World() = delete;
    World(World const &) = delete;
    World(World &&) = delete;
    World & operator=(World const &) = delete;
    World & operator=(World &&) = delete;
    ~World() = default;

    void add_vertex(vertex_type const & vertex);
    void add_vertex(value_type x, value_type y, value_type z)
    {
        add_vertex(vertex_type(x, y, z));
    }
    size_t nvertex() const { return m_vertices.size(); }
    vertex_type const & vertex(size_t i) const { return m_vertices[i]; }
    vertex_type & vertex(size_t i) { return m_vertices[i]; }
    vertex_type const & vertex_at(size_t i) const
    {
        check_size(i, m_vertices.size(), "vertex");
        return m_vertices[i];
    }
    vertex_type & vertex_at(size_t i)
    {
        check_size(i, m_vertices.size(), "vertex");
        return m_vertices[i];
    }

    void add_edge(edge_type const & edge);
    void add_edge(value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
    {
        add_edge(edge_type(x0, y0, z0, x1, y1, z1));
    }
    size_t nedge() const { return m_edges.size(); }
    edge_type const & edge(size_t i) const { return m_edges[i]; }
    edge_type & edge(size_t i) { return m_edges[i]; }
    edge_type const & edge_at(size_t i) const
    {
        check_size(i, m_edges.size(), "edge");
        return m_edges[i];
    }
    edge_type & edge_at(size_t i)
    {
        check_size(i, m_edges.size(), "edge");
        return m_edges[i];
    }

    void add_bezier(std::vector<vector_type> const & controls);
    size_t nbezier() const { return m_beziers.size(); }
    bezier_type const & bezier(size_t i) const { return m_beziers[i]; }
    bezier_type & bezier(size_t i) { return m_beziers[i]; }
    bezier_type const & bezier_at(size_t i) const
    {
        check_size(i, m_beziers.size(), "bezier");
        return m_beziers[i];
    }
    bezier_type & bezier_at(size_t i)
    {
        check_size(i, m_beziers.size(), "bezier");
        return m_beziers[i];
    }

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "World: (" << msg << ") i " << i << " >= size " << s);
        }
    }

    SimpleCollector<vertex_type> m_vertices;
    std::deque<Edge3d<T>> m_edges;
    std::deque<Bezier3d<T>> m_beziers;

}; /* end class World */

template <typename T>
void World<T>::add_vertex(vertex_type const & vertex)
{
    m_vertices.push_back(vertex);
}

template <typename T>
void World<T>::add_edge(edge_type const & edge)
{
    m_edges.emplace_back(edge);
}

template <typename T>
void World<T>::add_bezier(std::vector<vector_type> const & controls)
{
    m_beziers.emplace_back(controls);
}

using WorldFp32 = World<float>;
using WorldFp64 = World<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
