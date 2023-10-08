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

namespace modmesh
{

/**
 * Vector or point in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Vector3d
{

public:

    using value_type = T;

    Vector3d(T x, T y, T z)
        : m_coord{x, y, z}
    {
    }

    Vector3d() = default;
    Vector3d(Vector3d const &) = default;
    Vector3d(Vector3d &&) = default;
    Vector3d & operator=(Vector3d const &) = default;
    Vector3d & operator=(Vector3d &&) = default;
    ~Vector3d() = default;

    value_type x() const { return m_coord[0]; }
    value_type & x() { return m_coord[0]; }
    void set_x(value_type v) { x() = v; }

    value_type y() const { return m_coord[1]; }
    value_type & y() { return m_coord[1]; }
    void set_y(value_type v) { y() = v; }

    value_type z() const { return m_coord[2]; }
    value_type & z() { return m_coord[2]; }
    void set_z(value_type v) { z() = v; }

    T operator[](size_t i) const { return m_coord[i]; }
    T & operator[](size_t i) { return m_coord[i]; }

    T at(size_t i) const
    {
        check_size(i, 3);
        return m_coord[i];
    }
    T & at(size_t i)
    {
        check_size(i, 3);
        return m_coord[i];
    }

    size_t size() const { return 3; }

    void fill(T v) { m_coord[0] = m_coord[1] = m_coord[2] = v; }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Vector3d: i " << i << " >= size " << s);
        }
    }

    T m_coord[3];

}; /* end class Vector3d */

using Vector3dFp32 = Vector3d<float>;
using Vector3dFp64 = Vector3d<double>;

/**
 * Bezier curve in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Bezier3d
{

public:

    using vector_type = Vector3d<T>;

    Bezier3d(std::vector<vector_type> const & controls)
        : m_controls(controls)
    {
    }

    Bezier3d() = default;
    Bezier3d(Bezier3d const &) = default;
    Bezier3d(Bezier3d &&) = default;
    Bezier3d & operator=(Bezier3d const &) = default;
    Bezier3d & operator=(Bezier3d &&) = default;
    ~Bezier3d() = default;

    size_t size() const { return ncontrol(); }
    vector_type const & operator[](size_t i) const { return control(i); }
    vector_type & operator[](size_t i) { return control(i); }
    vector_type const & at(size_t i) const { return control_at(i); }
    vector_type & at(size_t i) { return control_at(i); }

    size_t ncontrol() const { return m_controls.size(); }
    vector_type const & control(size_t i) const { return m_controls[i]; }
    vector_type & control(size_t i) { return m_controls[i]; }
    vector_type const & control_at(size_t i) const
    {
        check_size(i, m_controls.size(), "control");
        return m_controls[i];
    }
    vector_type & control_at(size_t i)
    {
        check_size(i, m_controls.size(), "control");
        return m_controls[i];
    }

    size_t nlocus() const { return m_loci.size(); }
    vector_type const & locus(size_t i) const
    {
        check_size(i, m_loci.size(), "locus");
        return m_loci[i];
    }
    vector_type & locus(size_t i)
    {
        check_size(i, m_loci.size(), "locus");
        return m_loci[i];
    }

    void sample(size_t nlocus);

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Bezier3d: (" << msg << ") i " << i << " >= size " << s);
        }
    }

    std::vector<vector_type> m_controls;
    std::vector<vector_type> m_loci;

}; /* end class Bezier3d */

template <typename T>
void Bezier3d<T>::sample(size_t nlocus)
{
    if (nlocus < 2)
    {
        throw std::invalid_argument(Formatter() << "Bezier3d::sample: nlocus " << nlocus << " < 2");
    }
    m_loci.resize(nlocus);
    for (size_t idim = 0; idim < 3; ++idim)
    {
        std::vector<double> cvalues(ncontrol());
        for (size_t i = 0; i < ncontrol(); ++i)
        {
            cvalues[i] = control(i)[idim];
        }
        for (size_t i = 0; i < nlocus; ++i)
        {
            double const t = ((double)i) / (nlocus - 1);
            double const v = interpolate_bernstein(t, cvalues, cvalues.size() - 1);
            m_loci[i][idim] = v;
        }
    }
}

using Bezier3dFp32 = Bezier3d<float>;
using Bezier3dFp64 = Bezier3d<double>;

/**
 * Placeholder class for all geometry entities.
 */
class World
{

public:
private:

}; /* end class World */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
