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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: