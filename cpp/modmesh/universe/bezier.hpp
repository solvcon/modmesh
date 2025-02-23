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
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/universe/bernstein.hpp>

namespace modmesh
{

// TODO: Vector3d should be renamed as Point3d
/**
 * Point in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Vector3d
    : public NumberBase<int32_t, T>
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

template <typename T>
class PointPad
    : public NumberBase<int32_t, T>
{

public:

    using real_type = T;
    using value_type = T;
    using vector_type = Vector3d<T>;

    PointPad() = delete;

    PointPad(uint8_t ndim)
        : m_ndim(ndim)
    {
        if (ndim > 3)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " > 3");
        }
        else if (ndim < 2)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " < 2");
        }
    }

    PointPad(uint8_t ndim, size_t nelem)
        : m_ndim(ndim)
        , m_x(nelem)
        , m_y(nelem)
        , m_z()
    {
        if (ndim == 3)
        {
            m_z.expand(nelem);
        }
        else if (ndim > 3)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " > 3");
        }
        else if (ndim < 2)
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "ndim = " << int(ndim) << " < 2");
        }
    }

    PointPad(SimpleArray<T> & x, SimpleArray<T> & y, bool clone)
        : m_ndim(2)
        , m_x(x, clone)
        , m_y(y, clone)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size()
                << " are not the same");
        }
    }

    PointPad(SimpleArray<T> & x, SimpleArray<T> & y, SimpleArray<T> & z, bool clone)
        : m_ndim(3)
        , m_x(x, clone)
        , m_y(y, clone)
        , m_z(z, clone)
    {
        if (x.size() != y.size() || x.size() != z.size() || y.size() != z.size())
        {
            throw std::invalid_argument(
                Formatter()
                << "PointPad::PointPad: "
                << "x.size() " << x.size() << " y.size() " << y.size() << " z.size() " << z.size()
                << " are not the same");
        }
    }

    PointPad(PointPad const &) = default;
    PointPad(PointPad &&) = default;
    PointPad & operator=(PointPad const &) = default;
    PointPad & operator=(PointPad &&) = default;

    ~PointPad() = default;

    void append(T x, T y)
    {
        if (m_ndim != 2)
        {
            throw std::out_of_range(Formatter() << "PointPad::append: ndim must be 2 but is " << int(m_ndim));
        }
        m_x.push_back(x);
        m_y.push_back(y);
    }

    void append(T x, T y, T z)
    {
        if (m_ndim != 3)
        {
            throw std::out_of_range(Formatter() << "PointPad::append: ndim must be 3 but is " << int(m_ndim));
        }
        m_x.push_back(x);
        m_y.push_back(y);
        m_z.push_back(z);
    }

    // Do not implement setter of m_ndim. It should not be changed after construction.
    uint8_t ndim() const { return m_ndim; }

    size_t size() const { return m_x.size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{m_x.size(), m_ndim});
        if (m_ndim == 3)
        {
            for (size_t i = 0; i < m_x.size(); ++i)
            {
                ret(i, 0) = m_x[i];
                ret(i, 1) = m_y[i];
                ret(i, 2) = m_z[i];
            }
        }
        else
        {
            for (size_t i = 0; i < m_x.size(); ++i)
            {
                ret(i, 0) = m_x[i];
                ret(i, 1) = m_y[i];
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_x.expand(length);
        m_y.expand(length);
        if (m_ndim == 3)
        {
            m_z.expand(length);
        }
    }

    real_type x_at(size_t i) const { return m_x.at(i); }
    real_type y_at(size_t i) const { return m_y.at(i); }
    real_type z_at(size_t i) const { return m_z.at(i); }
    real_type & x_at(size_t i) { return m_x.at(i); }
    real_type & y_at(size_t i) { return m_y.at(i); }
    real_type & z_at(size_t i) { return m_z.at(i); }

    real_type x(size_t i) const { return m_x[i]; }
    real_type y(size_t i) const { return m_y[i]; }
    real_type z(size_t i) const { return m_z[i]; }
    real_type & x(size_t i) { return m_x[i]; }
    real_type & y(size_t i) { return m_y[i]; }
    real_type & z(size_t i) { return m_z[i]; }

    SimpleArray<value_type> x() { return m_x.as_array(); }
    SimpleArray<value_type> y() { return m_y.as_array(); }
    SimpleArray<value_type> z() { return m_z.as_array(); }

    vector_type get_at(size_t i) const
    {
        if (m_ndim == 3)
        {
            return vector_type(m_x.at(i), m_y.at(i), m_z.at(i));
        }
        else
        {
            return vector_type(m_x.at(i), m_y.at(i), 0.0);
        }
    }
    void set_at(size_t i, vector_type const & v)
    {
        m_x.at(i) = v.x();
        m_y.at(i) = v.y();
        if (m_ndim == 3)
        {
            m_z.at(i) = v.z();
        }
    }
    void set_at(size_t i, value_type x, value_type y)
    {
        m_x.at(i) = x;
        m_y.at(i) = y;
    }
    void set_at(size_t i, value_type x, value_type y, value_type z)
    {
        m_x.at(i) = x;
        m_y.at(i) = y;
        if (m_ndim == 3)
        {
            m_z.at(i) = z;
        }
    }

    vector_type get(size_t i) const
    {
        if (m_ndim == 3)
        {
            return vector_type(m_x[i], m_y[i], m_z[i]);
        }
        else
        {
            return vector_type(m_x[i], m_y[i], 0.0);
        }
    }
    void set(size_t i, vector_type const & v)
    {
        m_x[i] = v.x();
        m_y[i] = v.y();
        if (m_ndim == 3)
        {
            m_z[i] = v.z();
        }
    }
    void set(size_t i, value_type x, value_type y)
    {
        m_x[i] = x;
        m_y[i] = y;
    }
    void set(size_t i, value_type x, value_type y, value_type z)
    {
        m_x[i] = x;
        m_y[i] = y;
        if (m_ndim == 3)
        {
            m_z[i] = z;
        }
    }

private:

    uint8_t m_ndim;
    SimpleCollector<value_type> m_x;
    SimpleCollector<value_type> m_y;
    // For 2D point pads, m_z should remain unused and empty.
    SimpleCollector<value_type> m_z;

}; /* end class PointPad */

using PointPadFp32 = PointPad<float>;
using PointPadFp64 = PointPad<double>;

// TODO: Edge3d should be renamed as Segment3d
/**
 * Segment in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Edge3d
    : public NumberBase<int32_t, T>
{

public:

    using vector_type = Vector3d<T>;
    using value_type = typename vector_type::value_type;

    Edge3d(vector_type const & v0, vector_type const & v1)
        : m_vec{v0, v1}
    {
    }

    Edge3d(value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1)
        : m_vec{vector_type{x0, y0, z0}, vector_type{x1, y1, z1}}
    {
    }

    Edge3d() = default;
    Edge3d(Edge3d const &) = default;
    Edge3d & operator=(Edge3d const &) = default;
    Edge3d(Edge3d &&) = default;
    Edge3d & operator=(Edge3d &&) = default;
    ~Edge3d() = default;

    vector_type const & v0() const { return m_vec[0]; }
    vector_type & v0() { return m_vec[0]; }
    void set_v0(vector_type const & v) { m_vec[0] = v; }
    vector_type const & v1() const { return m_vec[1]; }
    vector_type & v1() { return m_vec[1]; }
    void set_v1(vector_type const & v) { m_vec[1] = v; }

    value_type x0() const { return m_vec[0].x(); }
    value_type & x0() { return m_vec[0].x(); }
    void set_x0(value_type v) { m_vec[0].set_x(v); }

    value_type y0() const { return m_vec[0].y(); }
    value_type & y0() { return m_vec[0].y(); }
    void set_y0(value_type v) { m_vec[0].set_y(v); }

    value_type z0() const { return m_vec[0].z(); }
    value_type & z0() { return m_vec[0].z(); }
    void set_z0(value_type v) { m_vec[0].set_z(v); }

    value_type x1() const { return m_vec[1].x(); }
    value_type & x1() { return m_vec[1].x(); }
    void set_x1(value_type v) { m_vec[1].set_x(v); }

    value_type y1() const { return m_vec[1].y(); }
    value_type & y1() { return m_vec[1].y(); }
    void set_y1(value_type v) { m_vec[1].set_y(v); }

    value_type z1() const { return m_vec[1].z(); }
    value_type & z1() { return m_vec[1].z(); }
    void set_z1(value_type v) { m_vec[1].set_z(v); }

    vector_type const & operator[](size_t i) const { return m_vec[i]; }
    vector_type & operator[](size_t i) { return m_vec[i]; }

    vector_type const & at(size_t i) const
    {
        check_size(i, 2);
        return m_vec[i];
    }
    vector_type & at(size_t i)
    {
        check_size(i, 2);
        return m_vec[i];
    }

    size_t size() const { return 2; }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Edge3d: i " << i << " >= size " << s);
        }
    }

    vector_type m_vec[2];

}; /* end class Edge3d */

using Edge3dFp32 = Edge3d<float>;
using Edge3dFp64 = Edge3d<double>;

/**
 * Bezier curve in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Bezier3d
    : public NumberBase<int32_t, T>
{

public:

    using vector_type = Vector3d<T>;

    explicit Bezier3d(std::vector<vector_type> const & controls)
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
        std::vector<T> cvalues(ncontrol());
        for (size_t i = 0; i < ncontrol(); ++i)
        {
            cvalues[i] = control(i)[idim];
        }
        for (size_t i = 0; i < nlocus; ++i)
        {
            T const t = ((T)i) / (nlocus - 1);
            T const v = detail::interpolate_bernstein_impl(t, cvalues, cvalues.size() - 1);
            m_loci[i][idim] = v;
        }
    }
}

using Bezier3dFp32 = Bezier3d<float>;
using Bezier3dFp64 = Bezier3d<double>;

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
