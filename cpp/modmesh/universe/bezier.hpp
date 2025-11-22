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
#include <modmesh/universe/coord.hpp>

namespace modmesh
{

namespace detail
{

template <typename T>
struct Triangle3dNamed
{
    T x0, x1, x2, y0, y1, y2, z0, z1, z2;
}; /* end struct Triangle3dNamed */

template <typename T>
union Triangle3dData
{
    T v[9];
    Triangle3dNamed<T> f;
}; /* end union Triangle3dData */

} /* end namespace detail */

/**
 * Triangle in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Triangle3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_arithmetic_v<T>, "T in Triangle3d<T> must be arithmetic type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Triangle3d(point_type const & p0, point_type const & p1, point_type const & p2)
        : m_data{p0.x(), p1.x(), p2.x(), p0.y(), p1.y(), p2.y(), p0.z(), p1.z(), p2.z()}
    {
    }

    Triangle3d() = default;
    Triangle3d(Triangle3d const &) = default;
    Triangle3d & operator=(Triangle3d const &) = default;
    Triangle3d(Triangle3d &&) = default;
    Triangle3d & operator=(Triangle3d &&) = default;
    ~Triangle3d() = default;

    point_type p0() const { return point_type(m_data.f.x0, m_data.f.y0, m_data.f.z0); }
    void set_p0(point_type const & p)
    {
        m_data.f.x0 = p.x();
        m_data.f.y0 = p.y();
        m_data.f.z0 = p.z();
    }
    point_type p1() const { return point_type(m_data.f.x1, m_data.f.y1, m_data.f.z1); }
    void set_p1(point_type const & p)
    {
        m_data.f.x1 = p.x();
        m_data.f.y1 = p.y();
        m_data.f.z1 = p.z();
    }
    point_type p2() const { return point_type(m_data.f.x2, m_data.f.y2, m_data.f.z2); }
    void set_p2(point_type const & p)
    {
        m_data.f.x2 = p.x();
        m_data.f.y2 = p.y();
        m_data.f.z2 = p.z();
    }

    value_type x0() const { return m_data.f.x0; }
    value_type & x0() { return m_data.f.x0; }
    void set_x0(value_type v) { m_data.f.x0 = v; }

    value_type y0() const { return m_data.f.y0; }
    value_type & y0() { return m_data.f.y0; }
    void set_y0(value_type v) { m_data.f.y0 = v; }

    value_type z0() const { return m_data.f.z0; }
    value_type & z0() { return m_data.f.z0; }
    void set_z0(value_type v) { m_data.f.z0 = v; }

    value_type x1() const { return m_data.f.x1; }
    value_type & x1() { return m_data.f.x1; }
    void set_x1(value_type v) { m_data.f.x1 = v; }

    value_type y1() const { return m_data.f.y1; }
    value_type & y1() { return m_data.f.y1; }
    void set_y1(value_type v) { m_data.f.y1 = v; }

    value_type z1() const { return m_data.f.z1; }
    value_type & z1() { return m_data.f.z1; }
    void set_z1(value_type v) { m_data.f.z1 = v; }

    value_type x2() const { return m_data.f.x2; }
    value_type & x2() { return m_data.f.x2; }
    void set_x2(value_type v) { m_data.f.x2 = v; }

    value_type y2() const { return m_data.f.y2; }
    value_type & y2() { return m_data.f.y2; }
    void set_y2(value_type v) { m_data.f.y2 = v; }

    value_type z2() const { return m_data.f.z2; }
    value_type & z2() { return m_data.f.z2; }
    void set_z2(value_type v) { m_data.f.z2 = v; }

    point_type operator[](size_t i) const { return point_type(m_data.v[i], m_data.v[i + 3], m_data.v[i + 6]); }

    point_type at(size_t i) const
    {
        check_size(i, 3);
        return (*this)[i];
    }

    bool operator==(Triangle3d const & other) const
    {
        return m_data.v[0] == other.m_data.v[0] &&
               m_data.v[1] == other.m_data.v[1] &&
               m_data.v[2] == other.m_data.v[2] &&
               m_data.v[3] == other.m_data.v[3] &&
               m_data.v[4] == other.m_data.v[4] &&
               m_data.v[5] == other.m_data.v[5] &&
               m_data.v[6] == other.m_data.v[6] &&
               m_data.v[7] == other.m_data.v[7] &&
               m_data.v[8] == other.m_data.v[8];
    }

    bool operator!=(Triangle3d const & other) const
    {
        return m_data.v[0] != other.m_data.v[0] ||
               m_data.v[1] != other.m_data.v[1] ||
               m_data.v[2] != other.m_data.v[2] ||
               m_data.v[3] != other.m_data.v[3] ||
               m_data.v[4] != other.m_data.v[4] ||
               m_data.v[5] != other.m_data.v[5] ||
               m_data.v[6] != other.m_data.v[6] ||
               m_data.v[7] != other.m_data.v[7] ||
               m_data.v[8] != other.m_data.v[8];
    }

    size_t size() const { return 3; }

    /**
     * Mirror the triangle with respect to the X axis.
     * This negates Y and Z coordinates, keeping X unchanged.
     */
    void mirror_x()
    {
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
    }

    /**
     * Mirror the triangle with respect to the Y axis.
     * This negates X and Z coordinates, keeping Y unchanged.
     */
    void mirror_y()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
    }

    /**
     * Mirror the triangle with respect to the Z axis.
     * This negates X and Y coordinates, keeping Z unchanged.
     */
    void mirror_z()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Triangle3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Triangle3d: i " << i << " >= size " << s);
        }
    }

    detail::Triangle3dData<T> m_data;

}; /* end class Triangle3d */

using Triangle3dFp32 = Triangle3d<float>;
using Triangle3dFp64 = Triangle3d<double>;

/// TrianglePad class for storing multiple Triangle3d objects
template <typename T>
class TrianglePad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<TrianglePad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using triangle_type = Triangle3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<TrianglePad<T>> construct(Args &&... args)
    {
        return std::make_shared<TrianglePad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    TrianglePad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
        , m_p2(point_pad_type::construct(ndim))
    {
    }

    TrianglePad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
        , m_p2(point_pad_type::construct(ndim, nelem))
    {
    }

    TrianglePad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0))
        , m_p1(point_pad_type::construct(x1, y1))
        , m_p2(point_pad_type::construct(x2, y2))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & z0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & z1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        SimpleArray<T> const & z2,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0))
        , m_p1(point_pad_type::construct(x1, y1, z1))
        , m_p2(point_pad_type::construct(x2, y2, z2))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, clone))
        , m_p1(point_pad_type::construct(x1, y1, clone))
        , m_p2(point_pad_type::construct(x2, y2, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & z0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & z1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        SimpleArray<T> & z2,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0, clone))
        , m_p1(point_pad_type::construct(x1, y1, z1, clone))
        , m_p2(point_pad_type::construct(x2, y2, z2, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    std::shared_ptr<TrianglePad<T>> clone()
    {
        if (ndim() == 2)
        {
            return TrianglePad<T>::construct(x0(), y0(), x1(), y1(), x2(), y2());
        }
        else
        {
            return TrianglePad<T>::construct(x0(), y0(), z0(), x1(), y1(), z1(), x2(), y2(), z2());
        }
    }

    TrianglePad() = delete;
    TrianglePad(TrianglePad const &) = delete;
    TrianglePad(TrianglePad &&) = delete;
    TrianglePad & operator=(TrianglePad const &) = delete;
    TrianglePad & operator=(TrianglePad &&) = delete;

    ~TrianglePad() = default;

    void append(triangle_type const & t)
    {
        if (ndim() == 2)
        {
            m_p0->append(t.x0(), t.y0());
            m_p1->append(t.x1(), t.y1());
            m_p2->append(t.x2(), t.y2());
        }
        else
        {
            m_p0->append(t.x0(), t.y0(), t.z0());
            m_p1->append(t.x1(), t.y1(), t.z1());
            m_p2->append(t.x2(), t.y2(), t.z2());
        }
    }

    void append(point_type const & p0, point_type const & p1, point_type const & p2)
    {
        if (ndim() == 2)
        {
            m_p0->append(p0.x(), p0.y());
            m_p1->append(p1.x(), p1.y());
            m_p2->append(p2.x(), p2.y());
        }
        else
        {
            m_p0->append(p0.x(), p0.y(), p0.z());
            m_p1->append(p1.x(), p1.y(), p1.z());
            m_p2->append(p2.x(), p2.y(), p2.z());
        }
    }

    void append(T x0, T y0, T x1, T y1, T x2, T y2)
    {
        m_p0->append(x0, y0);
        m_p1->append(x1, y1);
        m_p2->append(x2, y2);
    }

    void append(T x0, T y0, T z0, T x1, T y1, T z1, T x2, T y2, T z2)
    {
        m_p0->append(x0, y0, z0);
        m_p1->append(x1, y1, z1);
        m_p2->append(x2, y2, z2);
    }

    void extend_with(TrianglePad<T> const & other)
    {
        size_t const ntri = other.size();
        for (size_t i = 0; i < ntri; ++i)
        {
            append(other.get(i));
        }
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
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

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
        m_p2->expand(length);
    }

    real_type x0_at(size_t i) const { return m_p0->x_at(i); }
    real_type y0_at(size_t i) const { return m_p0->y_at(i); }
    real_type z0_at(size_t i) const { return m_p0->z_at(i); }
    real_type x1_at(size_t i) const { return m_p1->x_at(i); }
    real_type y1_at(size_t i) const { return m_p1->y_at(i); }
    real_type z1_at(size_t i) const { return m_p1->z_at(i); }
    real_type x2_at(size_t i) const { return m_p2->x_at(i); }
    real_type y2_at(size_t i) const { return m_p2->y_at(i); }
    real_type z2_at(size_t i) const { return m_p2->z_at(i); }
    real_type & x0_at(size_t i) { return m_p0->x_at(i); }
    real_type & y0_at(size_t i) { return m_p0->y_at(i); }
    real_type & z0_at(size_t i) { return m_p0->z_at(i); }
    real_type & x1_at(size_t i) { return m_p1->x_at(i); }
    real_type & y1_at(size_t i) { return m_p1->y_at(i); }
    real_type & z1_at(size_t i) { return m_p1->z_at(i); }
    real_type & x2_at(size_t i) { return m_p2->x_at(i); }
    real_type & y2_at(size_t i) { return m_p2->y_at(i); }
    real_type & z2_at(size_t i) { return m_p2->z_at(i); }

    real_type x0(size_t i) const { return m_p0->x(i); }
    real_type y0(size_t i) const { return m_p0->y(i); }
    real_type z0(size_t i) const { return m_p0->z(i); }
    real_type x1(size_t i) const { return m_p1->x(i); }
    real_type y1(size_t i) const { return m_p1->y(i); }
    real_type z1(size_t i) const { return m_p1->z(i); }
    real_type x2(size_t i) const { return m_p2->x(i); }
    real_type y2(size_t i) const { return m_p2->y(i); }
    real_type z2(size_t i) const { return m_p2->z(i); }
    real_type & x0(size_t i) { return m_p0->x(i); }
    real_type & y0(size_t i) { return m_p0->y(i); }
    real_type & z0(size_t i) { return m_p0->z(i); }
    real_type & x1(size_t i) { return m_p1->x(i); }
    real_type & y1(size_t i) { return m_p1->y(i); }
    real_type & z1(size_t i) { return m_p1->z(i); }
    real_type & x2(size_t i) { return m_p2->x(i); }
    real_type & y2(size_t i) { return m_p2->y(i); }
    real_type & z2(size_t i) { return m_p2->z(i); }

    point_type p0_at(size_t i) const { return m_p0->get_at(i); }
    point_type p1_at(size_t i) const { return m_p1->get_at(i); }
    point_type p2_at(size_t i) const { return m_p2->get_at(i); }
    void set_p0_at(size_t i, point_type const & p) { m_p0->set_at(i, p); }
    void set_p1_at(size_t i, point_type const & p) { m_p1->set_at(i, p); }
    void set_p2_at(size_t i, point_type const & p) { m_p2->set_at(i, p); }

    point_type p0(size_t i) const { return m_p0->get(i); }
    point_type p1(size_t i) const { return m_p1->get(i); }
    point_type p2(size_t i) const { return m_p2->get(i); }
    void set_p0(size_t i, point_type const & p) { m_p0->set(i, p); }
    void set_p1(size_t i, point_type const & p) { m_p1->set(i, p); }
    void set_p2(size_t i, point_type const & p) { m_p2->set(i, p); }

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }
    SimpleArray<value_type> x2() { return m_p2->x(); }
    SimpleArray<value_type> y2() { return m_p2->y(); }
    SimpleArray<value_type> z2() { return m_p2->z(); }

    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }
    std::shared_ptr<point_pad_type> p2() { return m_p2; }

    triangle_type get_at(size_t i) const
    {
        if (ndim() == 3)
        {
            return triangle_type(point_type(x0_at(i), y0_at(i), z0_at(i)), point_type(x1_at(i), y1_at(i), z1_at(i)), point_type(x2_at(i), y2_at(i), z2_at(i)));
        }
        else
        {
            return triangle_type(point_type(x0_at(i), y0_at(i), 0.0), point_type(x1_at(i), y1_at(i), 0.0), point_type(x2_at(i), y2_at(i), 0.0));
        }
    }
    void set_at(size_t i, triangle_type const & t)
    {
        x0_at(i) = t.x0();
        y0_at(i) = t.y0();
        x1_at(i) = t.x1();
        y1_at(i) = t.y1();
        x2_at(i) = t.x2();
        y2_at(i) = t.y2();
        if (ndim() == 3)
        {
            z0_at(i) = t.z0();
            z1_at(i) = t.z1();
            z2_at(i) = t.z2();
        }
    }
    void set_at(size_t i, point_type const & p0, point_type const & p1, point_type const & p2)
    {
        x0_at(i) = p0.x();
        y0_at(i) = p0.y();
        x1_at(i) = p1.x();
        y1_at(i) = p1.y();
        x2_at(i) = p2.x();
        y2_at(i) = p2.y();
        if (ndim() == 3)
        {
            z0_at(i) = p0.z();
            z1_at(i) = p1.z();
            z2_at(i) = p2.z();
        }
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
        if (ndim() == 3)
        {
            z0_at(i) = z0;
            z1_at(i) = z1;
            z2_at(i) = z2;
        }
    }

    triangle_type get(size_t i) const
    {
        if (ndim() == 3)
        {
            return triangle_type(point_type(x0(i), y0(i), z0(i)), point_type(x1(i), y1(i), z1(i)), point_type(x2(i), y2(i), z2(i)));
        }
        else
        {
            return triangle_type(point_type(x0(i), y0(i), 0.0), point_type(x1(i), y1(i), 0.0), point_type(x2(i), y2(i), 0.0));
        }
    }
    void set(size_t i, triangle_type const & t)
    {
        x0(i) = t.x0();
        y0(i) = t.y0();
        x1(i) = t.x1();
        y1(i) = t.y1();
        x2(i) = t.x2();
        y2(i) = t.y2();
        if (ndim() == 3)
        {
            z0(i) = t.z0();
            z1(i) = t.z1();
            z2(i) = t.z2();
        }
    }
    void set(size_t i, point_type const & p0, point_type const & p1, point_type const & p2)
    {
        x0(i) = p0.x();
        y0(i) = p0.y();
        x1(i) = p1.x();
        y1(i) = p1.y();
        x2(i) = p2.x();
        y2(i) = p2.y();
        if (ndim() == 3)
        {
            z0(i) = p0.z();
            z1(i) = p1.z();
            z2(i) = p2.z();
        }
    }
    void set(size_t i, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
    {
        x0(i) = x0;
        y0(i) = y0;
        x1(i) = x1;
        y1(i) = y1;
        x2(i) = x2;
        y2(i) = y2;
    }
    void set(size_t i, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
    {
        x0(i) = x0;
        y0(i) = y0;
        x1(i) = x1;
        y1(i) = y1;
        x2(i) = x2;
        y2(i) = y2;
        if (ndim() == 3)
        {
            z0(i) = z0;
            z1(i) = z1;
            z2(i) = z2;
        }
    }

    /**
     * Mirror the triangle pad with respect to the X axis.
     * This negates Y and Z coordinates, keeping X unchanged.
     * Handles both 2D and 3D triangles.
     */
    void mirror_x()
    {
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
            }
        }
    }

    /**
     * Mirror the triangle pad with respect to the Y axis.
     * This negates X and Z coordinates, keeping Y unchanged.
     * Handles both 2D and 3D triangles.
     */
    void mirror_y()
    {
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
            }
        }
    }

    /**
     * Mirror the triangle pad with respect to the Z axis.
     * This negates X and Y coordinates, keeping Z unchanged.
     * Only works for 3D triangles.
     *
     * @throw std::out_of_range if ndim is not 3
     */
    void mirror_z()
    {
        if (ndim() != 3)
        {
            throw std::out_of_range(
                Formatter() << "TrianglePad::mirror_z: cannot mirror Z axis for ndim " << int(ndim()));
        }
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
        }
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("TrianglePad::mirror: invalid axis"); break;
        }
    }

private:

    void check_constructor_point_size(point_pad_type const & p0, point_pad_type const & p1, point_pad_type const & p2)
    {
        if (m_p0->size() != m_p1->size() || m_p0->size() != m_p2->size())
        {
            throw std::invalid_argument(
                Formatter()
                << "TrianglePad::TrianglePad: "
                << "p0.size() " << p0.size() << " p1.size() " << p1.size() << " p2.size() " << p2.size()
                << " are not the same");
        }
    }

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;
    std::shared_ptr<point_pad_type> m_p2;

}; /* end class TrianglePad */

using TrianglePadFp32 = TrianglePad<float>;
using TrianglePadFp64 = TrianglePad<double>;

namespace detail
{

template <typename T>
struct Bezier3dNamed
{
    T x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
}; /* end struct Bezier3dNamed */

template <typename T>
union Bezier3dData
{
    T v[12];
    Bezier3dNamed<T> f;
}; /* end union Segment3dData */

} /* end namespace detail */

/**
 * Bezier curve up to degree 3 in three-dimensional space.
 *
 * @tparam T floating-point type
 */
template <typename T>
class Bezier3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_floating_point_v<T>, "T in Bezier3d<T> must be floating-point type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Bezier3d(point_type const & p0,
             point_type const & p1,
             point_type const & p2,
             point_type const & p3)
        : m_data{p0.x(), p1.x(), p2.x(), p3.x(), p0.y(), p1.y(), p2.y(), p3.y(), p0.z(), p1.z(), p2.z(), p3.z()}
    {
    }

    Bezier3d() = default;
    Bezier3d(Bezier3d const &) = default;
    Bezier3d(Bezier3d &&) = default;
    Bezier3d & operator=(Bezier3d const &) = default;
    Bezier3d & operator=(Bezier3d &&) = default;
    ~Bezier3d() = default;

#define DECL_VALUE_ACCESSOR(C, I)                     \
    value_type C##I() const { return m_data.f.C##I; } \
    value_type & C##I() { return m_data.f.C##I; }
    // clang-format off
    DECL_VALUE_ACCESSOR(x, 0)
    DECL_VALUE_ACCESSOR(x, 1)
    DECL_VALUE_ACCESSOR(x, 2)
    DECL_VALUE_ACCESSOR(x, 3)
    DECL_VALUE_ACCESSOR(y, 0)
    DECL_VALUE_ACCESSOR(y, 1)
    DECL_VALUE_ACCESSOR(y, 2)
    DECL_VALUE_ACCESSOR(y, 3)
    DECL_VALUE_ACCESSOR(z, 0)
    DECL_VALUE_ACCESSOR(z, 1)
    DECL_VALUE_ACCESSOR(z, 2)
    DECL_VALUE_ACCESSOR(z, 3)
    // clang-format on
#undef DECL_VALUE_ACCESSOR

#define DECL_POINT_ACCESSOR(I)                                             \
    point_type p##I() const { return point_type(x##I(), y##I(), z##I()); } \
    void set_p##I(point_type const & p)                                    \
    {                                                                      \
        x##I() = p.x();                                                    \
        y##I() = p.y();                                                    \
        z##I() = p.z();                                                    \
    }
    // clang-format off
    DECL_POINT_ACCESSOR(0)
    DECL_POINT_ACCESSOR(1)
    DECL_POINT_ACCESSOR(2)
    DECL_POINT_ACCESSOR(3)
    // clang-format on
#undef DECL_POINT_ACCESSOR

    std::shared_ptr<SegmentPad<T>> sample(size_t nlocus) const;

    /**
     * Mirror the Bezier curve with respect to the X axis.
     * This negates Y and Z coordinates, keeping X unchanged.
     */
    void mirror_x()
    {
        y0() = -y0();
        y1() = -y1();
        y2() = -y2();
        y3() = -y3();
        z0() = -z0();
        z1() = -z1();
        z2() = -z2();
        z3() = -z3();
    }

    /**
     * Mirror the Bezier curve with respect to the Y axis.
     * This negates X and Z coordinates, keeping Y unchanged.
     */
    void mirror_y()
    {
        x0() = -x0();
        x1() = -x1();
        x2() = -x2();
        x3() = -x3();
        z0() = -z0();
        z1() = -z1();
        z2() = -z2();
        z3() = -z3();
    }

    /**
     * Mirror the Bezier curve with respect to the Z axis.
     * This negates X and Y coordinates, keeping Z unchanged.
     */
    void mirror_z()
    {
        x0() = -x0();
        x1() = -x1();
        x2() = -x2();
        x3() = -x3();
        y0() = -y0();
        y1() = -y1();
        y2() = -y2();
        y3() = -y3();
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Bezier3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(Formatter() << "Bezier3d: (" << msg << ") i " << i << " >= size " << s);
        }
    }

    detail::Bezier3dData<T> m_data;

}; // namespace modmesh

using Bezier3dFp32 = Bezier3d<float>;
using Bezier3dFp64 = Bezier3d<double>;

/**
 * Store curves that are compatible to SVG
 * https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
 * @tparam T
 */
template <typename T>
class CurvePad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<CurvePad<T>>
{
private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using segment_type = Segment3d<T>;
    using bezier_type = Bezier3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<CurvePad<T>> construct(Args &&... args)
    {
        return std::make_shared<CurvePad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    CurvePad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
        , m_p2(point_pad_type::construct(ndim))
        , m_p3(point_pad_type::construct(ndim))
    {
    }

    CurvePad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
        , m_p2(point_pad_type::construct(ndim, nelem))
        , m_p3(point_pad_type::construct(ndim, nelem))
    {
    }

    CurvePad() = delete;
    CurvePad(CurvePad const &) = delete;
    CurvePad(CurvePad &&) = delete;
    CurvePad & operator=(CurvePad const &) = delete;
    CurvePad & operator=(CurvePad &&) = delete;

    ~CurvePad() = default;

    void append(bezier_type const & c)
    {
        m_p0->append(c.p0());
        m_p1->append(c.p1());
        m_p2->append(c.p2());
        m_p3->append(c.p3());
    }

    void append(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        m_p0->append(p0);
        m_p1->append(p1);
        m_p2->append(p2);
        m_p3->append(p3);
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{m_p0->size(), static_cast<size_t>(ndim() * 4)});
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
                ret(i, 9) = m_p3->x(i);
                ret(i, 10) = m_p3->y(i);
                ret(i, 11) = m_p3->z(i);
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
                ret(i, 6) = m_p3->x(i);
                ret(i, 7) = m_p3->y(i);
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
        m_p2->expand(length);
        m_p3->expand(length);
    }

#define DECL_VALUE_ACCESSOR(I, C)                                     \
    real_type C##I##_at(size_t i) const { return m_p##I->C##_at(i); } \
    real_type & C##I##_at(size_t i) { return m_p##I->C##_at(i); }     \
    real_type C##I(size_t i) const { return m_p##I->C(i); }           \
    real_type & C##I(size_t i) { return m_p##I->C(i); }
    // clang-format off
    DECL_VALUE_ACCESSOR(0, x)
    DECL_VALUE_ACCESSOR(0, y)
    DECL_VALUE_ACCESSOR(0, z)
    DECL_VALUE_ACCESSOR(1, x)
    DECL_VALUE_ACCESSOR(1, y)
    DECL_VALUE_ACCESSOR(1, z)
    DECL_VALUE_ACCESSOR(2, x)
    DECL_VALUE_ACCESSOR(2, y)
    DECL_VALUE_ACCESSOR(2, z)
    DECL_VALUE_ACCESSOR(3, x)
    DECL_VALUE_ACCESSOR(3, y)
    DECL_VALUE_ACCESSOR(3, z)
    // clang-format on
#undef DECL_VALUE_ACCESSOR

#define DECL_POINT_ACCESSOR(I)                                                          \
    point_type p##I##_at(size_t i) const { return m_p##I->get_at(i); }                  \
    void set_p##I##_at(size_t i, point_type const & p) { return m_p##I->set_at(i, p); } \
    point_type p##I(size_t i) const { return m_p##I->get(i); }                          \
    void set_p##I(size_t i, point_type const & p) { return m_p##I->set(i, p); }
    // clang-format off
    DECL_POINT_ACCESSOR(0)
    DECL_POINT_ACCESSOR(1)
    DECL_POINT_ACCESSOR(2)
    DECL_POINT_ACCESSOR(3)
    // clang-format on
#undef DECL_POINT_ACCESSOR

    bezier_type get_at(size_t i) const
    {
        return bezier_type(p0_at(i), p1_at(i), p2_at(i), p3_at(i));
    }
    void set_at(size_t i, bezier_type const & c)
    {
        m_p0->set_at(i, c.p0());
        m_p1->set_at(i, c.p1());
        m_p2->set_at(i, c.p2());
        m_p3->set_at(i, c.p3());
    }

    bezier_type get(size_t i) const
    {
        return bezier_type(p0(i), p1(i), p2(i), p3(i));
    }
    void set(size_t i, bezier_type const & c)
    {
        m_p0->set(i, c.p0());
        m_p1->set(i, c.p1());
        m_p2->set(i, c.p2());
        m_p3->set(i, c.p3());
    }

    // TODO: missing many accessors

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }
    SimpleArray<value_type> x2() { return m_p2->x(); }
    SimpleArray<value_type> y2() { return m_p2->y(); }
    SimpleArray<value_type> z2() { return m_p2->z(); }
    SimpleArray<value_type> x3() { return m_p3->x(); }
    SimpleArray<value_type> y3() { return m_p3->y(); }
    SimpleArray<value_type> z3() { return m_p3->z(); }

    // Should not implement the const versions of p0, p1, p2, p3 because the
    // returned shared pointers make the contents mutable.
    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }
    std::shared_ptr<point_pad_type> p2() { return m_p2; }
    std::shared_ptr<point_pad_type> p3() { return m_p3; }

    std::shared_ptr<SegmentPad<T>> sample(value_type length) const;

    /**
     * Mirror the curve pad with respect to the X axis.
     * This negates Y and Z coordinates of all control points, keeping X unchanged.
     */
    void mirror_x()
    {
        m_p0->mirror_x();
        m_p1->mirror_x();
        m_p2->mirror_x();
        m_p3->mirror_x();
    }

    /**
     * Mirror the curve pad with respect to the Y axis.
     * This negates X and Z coordinates of all control points, keeping Y unchanged.
     */
    void mirror_y()
    {
        m_p0->mirror_y();
        m_p1->mirror_y();
        m_p2->mirror_y();
        m_p3->mirror_y();
    }

    /**
     * Mirror the curve pad with respect to the Z axis.
     * This negates X and Y coordinates of all control points, keeping Z unchanged.
     */
    void mirror_z()
    {
        m_p0->mirror_z();
        m_p1->mirror_z();
        m_p2->mirror_z();
        m_p3->mirror_z();
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("CurvePad::mirror: invalid axis"); break;
        }
    }

private:

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;
    std::shared_ptr<point_pad_type> m_p2;
    std::shared_ptr<point_pad_type> m_p3;

}; /* end class CurvePad */

using CurvePadFp32 = CurvePad<float>;
using CurvePadFp64 = CurvePad<double>;

template <typename T>
class CubicBezierSampler
{
public:
    using point_type = Point3d<T>;
    using bezier_type = Bezier3d<T>;
    using curve_pad_type = CurvePad<T>;
    using segment_pad_type = SegmentPad<T>;

    using value_type = typename bezier_type::value_type;

    CubicBezierSampler(size_t ndim)
        : m_segments(segment_pad_type::construct(ndim))
    {
    }

    CubicBezierSampler() = delete;
    CubicBezierSampler(CubicBezierSampler const &) = delete;
    CubicBezierSampler(CubicBezierSampler &&) = default;
    CubicBezierSampler & operator=(CubicBezierSampler const &) = delete;
    CubicBezierSampler & operator=(CubicBezierSampler &&) = default;
    ~CubicBezierSampler() = default;

    std::shared_ptr<segment_pad_type> operator()(bezier_type const & curve, size_t nlocus, bool inplace);
    std::shared_ptr<segment_pad_type> operator()(curve_pad_type const & curves, T length);

    void reset() { m_segments.swap(segment_pad_type::construct(/*ndim*/ m_segments.ndim())); }

private:

    static size_t calc_nlocus(Bezier3d<T> const & c, T length);

    static size_t sample_to(bezier_type const & c, segment_pad_type & segment, size_t nlocus);

    std::shared_ptr<segment_pad_type> m_segments;
}; /* end class CubicBezierSampler */

template <typename T>
size_t CubicBezierSampler<T>::calc_nlocus(Bezier3d<T> const & c, T length)
{
    // The verbose code helps debugger stepping, but I did not check the assembly for performance
    point_type const & tp0 = c.p0();
    point_type const & tp3 = c.p3();
    point_type const vec = tp3 - tp0;
    value_type val = vec.calc_length();
    val /= length;
    val = std::floor(val);
    return val < 2 ? 2 : static_cast<size_t>(val);
}

template <typename T>
size_t CubicBezierSampler<T>::sample_to(bezier_type const & c, segment_pad_type & segments, size_t nlocus)
{
    point_type const & tp0 = c.p0();
    point_type const & tp1 = c.p1();
    point_type const & tp2 = c.p2();
    point_type const & tp3 = c.p3();
    point_type lastp = tp0;
    size_t nseg = 0;
    for (size_t j = 1; j < nlocus - 1; ++j)
    {
        value_type t = j;
        t /= nlocus - 1;
        point_type thisp;
        for (size_t idim = 0; idim < 3; ++idim)
        {
            std::vector<T> cvalues{tp0[idim], tp1[idim], tp2[idim], tp3[idim]};
            thisp[idim] = detail::interpolate_bernstein_impl(t, cvalues, cvalues.size() - 1);
        }
        segments.append(lastp, thisp);
        ++nseg;
        lastp = thisp;
    }
    segments.append(lastp, tp3);
    ++nseg;
    return nseg;
}

// Sample for all curves in a curve pad sequentially
template <typename T>
std::shared_ptr<SegmentPad<T>> CubicBezierSampler<T>::operator()(curve_pad_type const & curves, T length)
{
    for (size_t i = 0; i < curves.size(); ++i)
    {
        // Determine number of locus
        bezier_type const & c = curves.get(i);
        size_t const nlocus = calc_nlocus(c, length);
        // No sample and append the base line segment (p0-p3)
        if (nlocus <= 2)
        {
            m_segments->append(c.p0(), c.p3());
        }
        // Sample
        else
        {
            sample_to(c, *m_segments, nlocus);
        }
    }
    return m_segments;
}

template <typename T>
std::shared_ptr<SegmentPad<T>> CurvePad<T>::sample(value_type length) const
{
    return CubicBezierSampler<T>(/*ndim*/ 3)(*this, length);
}

// Sample for all curves in a curve pad sequentially
template <typename T>
std::shared_ptr<SegmentPad<T>> CubicBezierSampler<T>::operator()(bezier_type const & curve, size_t nlocus, bool inplace)
{
    std::shared_ptr<segment_pad_type> segments = inplace ? m_segments : segment_pad_type::construct(m_segments->ndim());
    // No sample and append the base line segment (p0-p3)
    if (nlocus <= 2)
    {
        segments->append(curve.p0(), curve.p3());
    }
    // Sample
    else
    {
        sample_to(curve, *segments, nlocus);
    }
    return segments;
}

template <typename T>
std::shared_ptr<SegmentPad<T>> Bezier3d<T>::sample(size_t nlocus) const
{
    return CubicBezierSampler<T>(/*ndim*/ 3)(*this, nlocus, /*inplace*/ false);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
