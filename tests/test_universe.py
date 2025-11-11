# Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import unittest

import numpy as np

import modmesh
from modmesh.testing import TestBase as ModMeshTB


class BernsteinTB(ModMeshTB):
    pass


class BernsteinPolynomialTC(BernsteinTB, unittest.TestCase):

    def test_degree1(self):
        # linear basis, degree n = 1
        f = modmesh.calc_bernstein_polynomial

        self.assertEqual(1.0, f(t=0.0, i=0, n=1))
        self.assertEqual(0.0, f(t=1.0, i=0, n=1))
        self.assertEqual(0.0, f(t=0.0, i=1, n=1))
        self.assertEqual(1.0, f(t=1.0, i=1, n=1))

        def _check(t):
            self.assert_allclose((1 - t), f(t, 0, 1))
            self.assert_allclose(t, f(t, 1, 1))
            vsum = f(t, 0, 1) + f(t, 1, 1)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)

    def test_degree2(self):
        # quadratic basis, degree n = 2
        f = modmesh.calc_bernstein_polynomial

        self.assert_allclose(1.0, f(t=0.0, i=0, n=2))
        self.assert_allclose(0.0, f(t=1.0, i=0, n=2))
        self.assert_allclose(0.0, f(t=0.0, i=2, n=2))
        self.assert_allclose(1.0, f(t=1.0, i=2, n=2))

        def _check(t):
            self.assert_allclose((1 - t) ** 2, f(t, 0, 2))
            self.assert_allclose(2 * (1 - t) * t, f(t, 1, 2))
            self.assert_allclose(t ** 2, f(t, 2, 2))
            vsum = f(t, 0, 2) + f(t, 1, 2) + f(t, 2, 2)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)

    def test_degree3(self):
        # cubic basis, degree n = 3
        f = modmesh.calc_bernstein_polynomial

        self.assertEqual(1.0, f(t=0.0, i=0, n=3))
        self.assertEqual(0.0, f(t=1.0, i=0, n=3))
        self.assertEqual(0.0, f(t=0.0, i=3, n=3))
        self.assertEqual(1.0, f(t=1.0, i=3, n=3))

        def _check(t):
            self.assert_allclose((1 - t) ** 3, f(t, 0, 3))
            self.assert_allclose(3 * ((1 - t) ** 2) * t, f(t, 1, 3))
            self.assert_allclose(3 * (1 - t) * (t ** 2), f(t, 2, 3))
            self.assert_allclose(t ** 3, f(t, 3, 3))
            vsum = f(t, 0, 3) + f(t, 1, 3) + f(t, 2, 3) + f(t, 3, 3)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)


class BernsteinInterpolationTC(BernsteinTB, unittest.TestCase):

    def test_degree1(self):
        # linear basis, degree n = 1
        f = modmesh.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) + values[1] * t,
            self.assert_allclose(golden, f(t=t, values=values, n=1))

        values = [1.0, 2.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=1))
        self.assertEqual(values[1], f(t=1.0, values=values, n=1))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)

    def test_degree2(self):
        # quadratic basis, degree n = 2
        f = modmesh.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) ** 2
            golden += values[1] * 2 * (1 - t) * t
            golden += values[2] * t ** 2
            self.assert_allclose(golden, f(t=t, values=values, n=2))

        values = [1.0, 2.0, 3.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=2))
        self.assertEqual(values[2], f(t=1.0, values=values, n=2))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)

    def test_degree3(self):
        # cubic basis, degree n = 3
        f = modmesh.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) ** 3
            golden += values[1] * 3 * ((1 - t) ** 2) * t
            golden += values[2] * 3 * (1 - t) * (t ** 2)
            golden += values[3] * t ** 3
            self.assert_allclose(golden, f(t=t, values=values, n=3))

        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=3))
        self.assertEqual(values[3], f(t=1.0, values=values, n=3))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)


class Point3dTB(ModMeshTB):

    def test_construct(self):
        Point = self.Point

        # Construct using positional arguments
        pnt = Point(1, 2, 3)
        self.assertEqual(pnt.x, 1.0)
        self.assertEqual(pnt.y, 2.0)
        self.assertEqual(pnt.z, 3.0)

        # Construct using keyword arguments
        pnt = Point(x=2.2, y=5.8, z=-9.22)
        self.assert_allclose(pnt, [2.2, 5.8, -9.22])
        self.assert_allclose(pnt[0], 2.2)
        self.assert_allclose(pnt[1], 5.8)
        self.assert_allclose(pnt[2], -9.22)
        self.assertEqual(len(pnt), 3)

        # Range error in C++
        with self.assertRaisesRegex(IndexError, "Point3d: i 3 >= size 3"):
            pnt[3]

    def test_fill(self):
        Point = self.Point

        pnt = Point(1, 2, 3)
        pnt.fill(10.0)
        self.assertEqual(list(pnt), [10, 10, 10])

    def test_arithmetic(self):
        Point = self.Point
        p1 = Point(1, 2, 3)
        p2 = Point(4, 5, 6)

        p1 += p2
        self.assertEqual(list(p1), [5, 7, 9])
        p1 -= p2
        self.assertEqual(list(p1), [1, 2, 3])

        p1 += 4
        self.assertEqual(list(p1), [5, 6, 7])
        p1 -= 2
        self.assertEqual(list(p1), [3, 4, 5])

        p1 *= 8
        self.assertEqual(list(p1), [24, 32, 40])
        p1 /= 4
        self.assertEqual(list(p1), [6, 8, 10])

    def test_mirror(self):
        Point = self.Point

        p1 = Point(1, 2, 3)
        p1.mirror(axis='x')
        self.assertEqual(list(p1), [1, -2, -3])

        p2 = Point(1, 2, 3)
        p2.mirror('y')
        self.assertEqual(list(p2), [-1, 2, -3])

        p3 = Point(1, 2, 3)
        p3.mirror('z')
        self.assertEqual(list(p3), [-1, -2, 3])

        p4 = Point(1, 2, 3)
        p4.mirror('X')
        self.assertEqual(list(p4), [1, -2, -3])

        with self.assertRaisesRegex(
                ValueError, "Point3d::mirror: axis must be 'x', 'y', or 'z'"):
            Point(1, 2, 3).mirror('w')


class Point3dFp32TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Point3dFp32, self.Point)

    def test_repr_str(self):
        from modmesh import Point3dFp32
        p = Point3dFp32(607.7, -64.2, 0)
        golden = "Point3dFp32(607.7, -64.2, 0)"
        # __repr__ is the same as __str__ for Point3d
        self.assertEqual(repr(p), golden)
        self.assertEqual(str(p), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        self.assertEqual(p, e)


class Point3dFp64TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Point3dFp64, self.Point)

    def test_repr_str(self):
        from modmesh import Point3dFp64
        p = Point3dFp64(607.7, -64.2, 0)
        golden = "Point3dFp64(607.7, -64.2, 0)"
        # __repr__ is the same as __str__ for Point3d
        self.assertEqual(repr(p), golden)
        self.assertEqual(str(p), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        self.assertEqual(p, e)


class Segment3dTB(ModMeshTB):

    def test_construct(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(p0=Point(x=0, y=0, z=0), p1=Point(x=1, y=1, z=1))
        self.assertEqual(len(s), 2)
        self.assertEqual(tuple(s.p0), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(s.p1), (1.0, 1.0, 1.0))

        s.p0 = Point(x=3, y=7, z=0)
        s.p1 = Point(x=-1, y=-4, z=9)
        self.assertEqual(s.x0, 3)
        self.assertEqual(s.y0, 7)
        self.assertEqual(s.z0, 0)
        self.assertEqual(s.x1, -1)
        self.assertEqual(s.y1, -4)
        self.assertEqual(s.z1, 9)

        s = Segment(Point(x=3.1, y=7.4, z=0.6), Point(x=-1.2, y=-4.1, z=9.2))
        self.assert_allclose(tuple(s.p0), (3.1, 7.4, 0.6))
        self.assert_allclose(tuple(s.p1), (-1.2, -4.1, 9.2))

    def test_mirror(self):
        Point = self.Point
        Segment = self.Segment

        s1 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s1.mirror('x')
        self.assertEqual(list(s1.p0), [1, -2, -3])
        self.assertEqual(list(s1.p1), [4, -5, -6])

        s2 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s2.mirror('y')
        self.assertEqual(list(s2.p0), [-1, 2, -3])
        self.assertEqual(list(s2.p1), [-4, 5, -6])

        s3 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s3.mirror('z')
        self.assertEqual(list(s3.p0), [-1, -2, 3])
        self.assertEqual(list(s3.p1), [-4, -5, 6])

        s4 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s4.mirror('Y')
        self.assertEqual(list(s4.p0), [-1, 2, -3])
        self.assertEqual(list(s4.p1), [-4, 5, -6])

        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::mirror: axis must be 'x', 'y', or 'z'"):
            Segment(Point(1, 2, 3), Point(4, 5, 6)).mirror('w')


class Segment3dFp32TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp32
        self.Segment = modmesh.Segment3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp32, Segment3dFp32
        s = Segment3dFp32(Point3dFp32(504.8, -64.2, 0),
                          Point3dFp32(421.4, -250.5, 0))
        golden = ("Segment3dFp32(Point3dFp32(504.8, -64.2, 0), "
                  "Point3dFp32(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Segment3d
        self.assertEqual(repr(s), golden)
        self.assertEqual(str(s), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        self.assertEqual(s, e)


class Segment3dFp64TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp64
        self.Segment = modmesh.Segment3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp64, Segment3dFp64
        s = Segment3dFp64(Point3dFp64(504.8, -64.2, 0),
                          Point3dFp64(421.4, -250.5, 0))
        golden = ("Segment3dFp64(Point3dFp64(504.8, -64.2, 0), "
                  "Point3dFp64(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Segment3d
        self.assertEqual(repr(s), golden)
        self.assertEqual(str(s), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        self.assertEqual(s, e)


class Triangle3dTB(ModMeshTB):

    def test_construct(self):
        Point = self.Point
        Triangle = self.Triangle

        t = Triangle(p0=Point(x=0, y=0, z=0),
                     p1=Point(x=1, y=1, z=1),
                     p2=Point(x=2, y=2, z=2))
        self.assertEqual(len(t), 3)
        self.assertEqual(tuple(t.p0), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(t.p1), (1.0, 1.0, 1.0))
        self.assertEqual(tuple(t.p2), (2.0, 2.0, 2.0))

        t.p0 = Point(x=3, y=7, z=0)
        t.p1 = Point(x=-1, y=-4, z=9)
        t.p2 = Point(x=5, y=6, z=-2)
        self.assertEqual(t.x0, 3)
        self.assertEqual(t.y0, 7)
        self.assertEqual(t.z0, 0)
        self.assertEqual(t.x1, -1)
        self.assertEqual(t.y1, -4)
        self.assertEqual(t.z1, 9)
        self.assertEqual(t.x2, 5)
        self.assertEqual(t.y2, 6)
        self.assertEqual(t.z2, -2)

        t = Triangle(Point(x=3.1, y=7.4, z=0.6),
                     Point(x=-1.2, y=-4.1, z=9.2),
                     Point(x=5.5, y=6.6, z=-2.3))
        self.assert_allclose(tuple(t.p0), (3.1, 7.4, 0.6))
        self.assert_allclose(tuple(t.p1), (-1.2, -4.1, 9.2))
        self.assert_allclose(tuple(t.p2), (5.5, 6.6, -2.3))

    def test_indexing(self):
        Point = self.Point
        Triangle = self.Triangle

        t = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        self.assertEqual(list(t[0]), [1, 2, 3])
        self.assertEqual(list(t[1]), [4, 5, 6])
        self.assertEqual(list(t[2]), [7, 8, 9])

        t.p0 = Point(10, 11, 12)
        self.assertEqual(list(t.p0), [10, 11, 12])
        t.p1 = Point(13, 14, 15)
        self.assertEqual(list(t.p1), [13, 14, 15])
        t.p2 = Point(16, 17, 18)
        self.assertEqual(list(t.p2), [16, 17, 18])

    def test_equality(self):
        Point = self.Point
        Triangle = self.Triangle

        t1 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t2 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t3 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 10))

        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_mirror(self):
        Point = self.Point
        Triangle = self.Triangle

        t1 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t1.mirror('x')
        self.assertEqual(list(t1.p0), [1, -2, -3])
        self.assertEqual(list(t1.p1), [4, -5, -6])
        self.assertEqual(list(t1.p2), [7, -8, -9])

        t2 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t2.mirror('y')
        self.assertEqual(list(t2.p0), [-1, 2, -3])
        self.assertEqual(list(t2.p1), [-4, 5, -6])
        self.assertEqual(list(t2.p2), [-7, 8, -9])

        t3 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t3.mirror('z')
        self.assertEqual(list(t3.p0), [-1, -2, 3])
        self.assertEqual(list(t3.p1), [-4, -5, 6])
        self.assertEqual(list(t3.p2), [-7, -8, 9])

        t4 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t4.mirror('Y')
        self.assertEqual(list(t4.p0), [-1, 2, -3])
        self.assertEqual(list(t4.p1), [-4, 5, -6])
        self.assertEqual(list(t4.p2), [-7, 8, -9])

        with self.assertRaisesRegex(
                ValueError,
                "Triangle3d::mirror: axis must be 'x', 'y', or 'z'"):
            Triangle(Point(1, 2, 3), Point(4, 5, 6),
                     Point(7, 8, 9)).mirror('w')


class Triangle3dFp32TC(Triangle3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp32
        self.Triangle = modmesh.Triangle3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp32, Triangle3dFp32
        t = Triangle3dFp32(Point3dFp32(1.5, 2.5, 3.5),
                           Point3dFp32(4.5, 5.5, 6.5),
                           Point3dFp32(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp32(Point3dFp32(1.5, 2.5, 3.5), "
                  "Point3dFp32(4.5, 5.5, 6.5), "
                  "Point3dFp32(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden)
        self.assertEqual(t, e)


class Triangle3dFp64TC(Triangle3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp64
        self.Triangle = modmesh.Triangle3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp64, Triangle3dFp64
        t = Triangle3dFp64(Point3dFp64(1.5, 2.5, 3.5),
                           Point3dFp64(4.5, 5.5, 6.5),
                           Point3dFp64(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp64(Point3dFp64(1.5, 2.5, 3.5), "
                  "Point3dFp64(4.5, 5.5, 6.5), "
                  "Point3dFp64(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden)
        self.assertEqual(t, e)


class Bezier3dTB(ModMeshTB):

    def test_control_points(self):
        Point = self.Point
        Bezier = self.Bezier

        # Create a cubic Bezier curve
        bzr = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0), p2=Point(3, 1, 0),
                     p3=Point(4, 0, 0))
        self.assertEqual(len(bzr), 4)
        self.assertEqual(list(bzr[0]), [0, 0, 0])
        self.assertEqual(list(bzr[1]), [1, 1, 0])
        self.assertEqual(list(bzr[2]), [3, 1, 0])
        self.assertEqual(list(bzr[3]), [4, 0, 0])

        # Range error in C++
        with self.assertRaisesRegex(IndexError,
                                    "Bezier3d: \\(control\\) i 4 >= size 4"):
            bzr[4]

    def test_locus_points(self):
        Point = self.Point
        Bezier = self.Bezier

        b = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0), p2=Point(3, 1, 0),
                   p3=Point(4, 0, 0))
        self.assertEqual(len(b), 4)

        segs = b.sample(nlocus=5)
        self.assertEqual(len(segs), 4)
        self.assert_allclose(
            list(segs[0]), [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[1]), [[0.90625, 0.5625, 0.0], [2.0, 0.75, 0.0]])
        self.assert_allclose(
            list(segs[2]), [[2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[3]), [[3.09375, 0.5625, 0.0], [4.0, 0.0, 0.0]])

        segs = b.sample(nlocus=9)
        self.assertEqual(len(segs), 8)
        self.assert_allclose(
            list(segs[0]), [[0.0, 0.0, 0.0], [0.41796875, 0.328125, 0.0]])
        self.assert_allclose(
            list(segs[1]),
            [[0.41796875, 0.328125, 0.0], [0.90625, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[2]),
            [[0.90625, 0.5625, 0.0], [1.44140625, 0.703125, 0.0]])
        self.assert_allclose(
            list(segs[3]), [[1.44140625, 0.703125, 0.0], [2.0, 0.75, 0.0]])
        self.assert_allclose(
            list(segs[4]), [[2.0, 0.75, 0.0], [2.55859375, 0.703125, 0.0]])
        self.assert_allclose(
            list(segs[5]),
            [[2.55859375, 0.703125, 0.0], [3.09375, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[6]),
            [[3.09375, 0.5625, 0.0], [3.58203125, 0.328125, 0.0]])
        self.assert_allclose(
            list(segs[7]), [[3.58203125, 0.328125, 0.0], [4.0, 0.0, 0.0]])

    def test_mirror(self):
        Point = self.Point
        Bezier = self.Bezier

        b1 = Bezier(Point(0, 0, 0), Point(1, 1, 0),
                    Point(3, 1, 0), Point(4, 0, 0))
        b1.mirror('x')
        self.assertEqual(list(b1[0]), [0, 0, 0])
        self.assertEqual(list(b1[1]), [1, -1, 0])
        self.assertEqual(list(b1[2]), [3, -1, 0])
        self.assertEqual(list(b1[3]), [4, 0, 0])

        b2 = Bezier(Point(0, 0, 0), Point(1, 1, 0),
                    Point(3, 1, 0), Point(4, 0, 0))
        b2.mirror('y')
        self.assertEqual(list(b2[0]), [0, 0, 0])
        self.assertEqual(list(b2[1]), [-1, 1, 0])
        self.assertEqual(list(b2[2]), [-3, 1, 0])
        self.assertEqual(list(b2[3]), [-4, 0, 0])

        b3 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b3.mirror('z')
        self.assertEqual(list(b3[0]), [-1, -2, 3])
        self.assertEqual(list(b3[1]), [-4, -5, 6])
        self.assertEqual(list(b3[2]), [-7, -8, 9])
        self.assertEqual(list(b3[3]), [-10, -11, 12])

        b4 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b4.mirror('Z')
        self.assertEqual(list(b4[0]), [-1, -2, 3])

        with self.assertRaisesRegex(
                ValueError,
                "Bezier3d::mirror: axis must be 'x', 'y', or 'z'"):
            Bezier(Point(0, 0, 0), Point(1, 1, 0),
                   Point(3, 1, 0), Point(4, 0, 0)).mirror('w')


class Bezier3dFp32TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp32
        self.Bezier = modmesh.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp32, Bezier3dFp32
        b = Bezier3dFp32(Point3dFp32(607.7, -64.2, 0),
                         Point3dFp32(504.8, -64.2, 0),
                         Point3dFp32(421.4, -147.6, 0),
                         Point3dFp32(421.4, -250.5, 0))
        golden = ("Bezier3dFp32(Point3dFp32(607.7, -64.2, 0), "
                  "Point3dFp32(504.8, -64.2, 0), "
                  "Point3dFp32(421.4, -147.6, 0), "
                  "Point3dFp32(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Bezier3d
        self.assertEqual(repr(b), golden)
        self.assertEqual(str(b), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        # FIXME: Bezier3d does not have equality operator
        # Tracked in https://github.com/solvcon/modmesh/issues/568
        self.assertEqual(b[0], e[0])
        self.assertEqual(b[1], e[1])
        self.assertEqual(b[2], e[2])
        self.assertEqual(b[3], e[3])


class Bezier3dFp64TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = modmesh.Point3dFp64
        self.Bezier = modmesh.Bezier3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        from modmesh import Point3dFp64, Bezier3dFp64
        b = Bezier3dFp64(Point3dFp64(607.7, -64.2, 0),
                         Point3dFp64(504.8, -64.2, 0),
                         Point3dFp64(421.4, -147.6, 0),
                         Point3dFp64(421.4, -250.5, 0))
        golden = ("Bezier3dFp64(Point3dFp64(607.7, -64.2, 0), "
                  "Point3dFp64(504.8, -64.2, 0), "
                  "Point3dFp64(421.4, -147.6, 0), "
                  "Point3dFp64(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Bezier3d
        self.assertEqual(repr(b), golden)
        self.assertEqual(str(b), golden)
        # Evaluate the string and test the result
        e = eval(golden)
        # FIXME: Bezier3d does not have equality operator
        # Tracked in https://github.com/solvcon/modmesh/issues/568
        self.assertEqual(b[0], e[0])
        self.assertEqual(b[1], e[1])
        self.assertEqual(b[2], e[2])
        self.assertEqual(b[3], e[3])


class PointPadTB(ModMeshTB):

    def test_ndim(self):
        pp2d = self.PointPad(ndim=2)
        self.assertEqual(pp2d.ndim, 2)
        pp3d = self.PointPad(ndim=3)
        self.assertEqual(pp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.PointPad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.PointPad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.PointPad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.PointPad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.PointPad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.PointPad(ndim=4, nelem=5)

    def test_construct_2d(self):
        xarr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        pp = self.PointPad(x=xarr, y=yarr, clone=False)
        self.assertEqual(pp.ndim, 2)
        self.assert_allclose(pp.x, [1, 2, 3])
        self.assert_allclose(pp.y, [4, 5, 6])
        self.assertEqual(len(pp.z), 0)

        # Test zero-copy writing
        pp.x[1] = 200.2
        pp.y[0] = -700.3
        self.assert_allclose(list(pp[0]), (1, -700.3, 0))
        self.assert_allclose(list(pp[1]), (200.2, 5, 0))
        self.assert_allclose(list(pp[2]), (3, 6, 0))

        pp2 = self.PointPad(ndim=2, nelem=3)
        for i in range(len(pp)):
            pp2.set_at(i, pp.get_at(i).x, pp.get_at(i).y)
        self.assert_allclose(pp2.x, [1, 200.2, 3])
        self.assert_allclose(pp2.y, [-700.3, 5, 6])
        self.assertEqual(len(pp2.z), 0)

        packed = pp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 2))
        self.assert_allclose(list(packed[0]), (1, -700.3))
        self.assert_allclose(list(packed[1]), (200.2, 5))
        self.assert_allclose(list(packed[2]), (3, 6))

    def test_construct_3d(self):
        xarr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        zarr = self.SimpleArray(array=np.array([7, 8, 9], dtype=self.dtype))
        pp = self.PointPad(x=xarr, y=yarr, z=zarr, clone=False)
        self.assertEqual(pp.ndim, 3)
        self.assert_allclose(pp.x, [1, 2, 3])
        self.assert_allclose(pp.y, [4, 5, 6])
        self.assert_allclose(pp.z, [7, 8, 9])

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.x_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.y_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.z_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.get_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.set_at(3, self.Point(0, 0, 0))

        # Test zero-copy writing
        pp.x[1] = 200.2
        pp.y[0] = -700.3
        pp.z[2] = 213.9
        self.assert_allclose(list(pp[0]), (1, -700.3, 7))
        self.assert_allclose(list(pp[1]), (200.2, 5, 8))
        self.assert_allclose(list(pp[2]), (3, 6, 213.9))

        pp2 = self.PointPad(ndim=3, nelem=3)
        for i in range(len(pp)):
            pp2.set_at(i, pp.get_at(i).x, pp.get_at(i).y, pp.get_at(i).z)
        self.assert_allclose(pp2.x, [1, 200.2, 3])
        self.assert_allclose(pp2.y, [-700.3, 5, 6])
        self.assert_allclose(pp2.z, [7, 8, 213.9])

        packed = pp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 3))
        self.assert_allclose(list(packed[0]), (1, -700.3, 7))
        self.assert_allclose(list(packed[1]), (200.2, 5, 8))
        self.assert_allclose(list(packed[2]), (3, 6, 213.9))

    def test_append_2d(self):
        pp = self.PointPad(ndim=2)
        self.assertEqual(pp.ndim, 2)
        self.assertEqual(len(pp), 0)
        pp.append(1.1, 2.2)
        self.assertEqual(len(pp), 1)
        self.assert_allclose(pp.x_at(0), 1.1)
        self.assert_allclose(pp.y_at(0), 2.2)
        pp.append(1.1 * 3, 2.2 * 3)
        self.assertEqual(len(pp), 2)
        self.assert_allclose(pp.x_at(1), 1.1 * 3)
        self.assert_allclose(pp.y_at(1), 2.2 * 3)
        pp.append(1.1 * 3.1, 2.2 * 3.1)
        self.assertEqual(len(pp), 3)
        self.assert_allclose(pp.x_at(2), 1.1 * 3.1)
        self.assert_allclose(pp.y_at(2), 2.2 * 3.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 3 but is 2"):
            pp.append(3.2, 4.1, 5.7)
        self.assertEqual(len(pp), 3)

        # Test batch interface
        self.assert_allclose(pp.x, [1.1, 1.1 * 3, 1.1 * 3.1])
        self.assert_allclose(pp.y, [2.2, 2.2 * 3, 2.2 * 3.1])
        pp.x[0] = -10.9
        pp.x.ndarray[2] = -13.2
        self.assert_allclose(pp.x_at(0), -10.9)
        self.assert_allclose(pp.x_at(1), 1.1 * 3)
        self.assert_allclose(pp.x_at(2), -13.2)
        pp.y[1] = -0.93
        pp.y.ndarray[2] = 29.1
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.y_at(1), -0.93)
        self.assert_allclose(pp.y_at(2), 29.1)
        self.assertEqual(len(pp.z), 0)

    def test_append_3d(self):
        pp = self.PointPad(ndim=3)
        self.assertEqual(pp.ndim, 3)
        self.assertEqual(len(pp), 0)
        pp.append(1.1, 2.2, 3.3)
        self.assertEqual(len(pp), 1)
        self.assert_allclose(pp.x_at(0), 1.1)
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.z_at(0), 3.3)
        pp.append(1.1 * 5, 2.2 * 5, 3.3 * 5)
        self.assertEqual(len(pp), 2)
        self.assert_allclose(pp.x_at(1), 1.1 * 5)
        self.assert_allclose(pp.y_at(1), 2.2 * 5)
        self.assert_allclose(pp.z_at(1), 3.3 * 5)
        pp.append(1.1 * 5.1, 2.2 * 5.1, 3.3 * 5.1)
        self.assertEqual(len(pp), 3)
        self.assert_allclose(pp.x_at(2), 1.1 * 5.1)
        self.assert_allclose(pp.y_at(2), 2.2 * 5.1)
        self.assert_allclose(pp.z_at(2), 3.3 * 5.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 2 but is 3"):
            pp.append(3.2, 4.1)
        self.assertEqual(len(pp), 3)

        # Test batch interface
        self.assert_allclose(pp.x, [1.1, 1.1 * 5, 1.1 * 5.1])
        self.assert_allclose(pp.y, [2.2, 2.2 * 5, 2.2 * 5.1])
        self.assert_allclose(pp.z, [3.3, 3.3 * 5, 3.3 * 5.1])
        pp.x[0] = -10.9
        pp.x.ndarray[2] = -13.2
        self.assert_allclose(pp.x_at(0), -10.9)
        self.assert_allclose(pp.x_at(1), 1.1 * 5)
        self.assert_allclose(pp.x_at(2), -13.2)
        pp.y[1] = -0.93
        pp.y.ndarray[2] = 29.1
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.y_at(1), -0.93)
        self.assert_allclose(pp.y_at(2), 29.1)
        pp.z[0] = 2.31
        pp.z.ndarray[1] = 8.23
        self.assert_allclose(pp.z_at(0), 2.31)
        self.assert_allclose(pp.z_at(1), 8.23)
        self.assert_allclose(pp.z_at(2), 3.3 * 5.1)

    def test_mirror_2d(self):
        PointPad = self.PointPad

        pp = PointPad(ndim=2)
        pp.append(1.0, 2.0)
        pp.append(3.0, 4.0)
        pp.append(5.0, 6.0)

        pp.mirror('x')
        self.assert_allclose(pp.x_at(0), 1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.x_at(1), 3.0)
        self.assert_allclose(pp.y_at(1), -4.0)

        pp.mirror('y')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.x_at(1), -3.0)
        self.assert_allclose(pp.y_at(1), -4.0)

    def test_mirror_3d(self):
        PointPad = self.PointPad

        pp = PointPad(ndim=3)
        pp.append(1.0, 2.0, 3.0)
        pp.append(4.0, 5.0, 6.0)

        pp.mirror('z')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.z_at(0), 3.0)
        self.assert_allclose(pp.x_at(1), -4.0)
        self.assert_allclose(pp.y_at(1), -5.0)
        self.assert_allclose(pp.z_at(1), 6.0)

        pp.mirror('X')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), 2.0)
        self.assert_allclose(pp.z_at(0), -3.0)
        self.assert_allclose(pp.x_at(1), -4.0)
        self.assert_allclose(pp.y_at(1), 5.0)
        self.assert_allclose(pp.z_at(1), -6.0)

        with self.assertRaisesRegex(
                ValueError, "PointPad::mirror: axis must be 'x', 'y', or 'z'"):
            pp.mirror('w')

    def test_alignment(self):
        # Test alignment property for PointPad with various configurations
        pp2d = self.PointPad(ndim=2)
        self.assertEqual(pp2d.alignment, 0)

        # Test alignment with valid values (16, 32, 64)
        pp2d_aligned = self.PointPad(ndim=2, nelem=8, alignment=32)
        self.assertEqual(pp2d_aligned.alignment, 32)

        pp3d_aligned = self.PointPad(ndim=3, nelem=16, alignment=64)
        self.assertEqual(pp3d_aligned.alignment, 64)

        # Test invalid alignment values
        invalid_alignments = [
            1, 2, 4, 8, 128, 256, 1024, 15, 17, 31, 33, 63, 65
        ]
        for alignment in invalid_alignments:
            with self.assertRaisesRegex(
                    ValueError, "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
                self.PointPad(ndim=2, nelem=4, alignment=alignment)

        # Test alignment with arrays and cloning
        xarr = self.SimpleArray(
            array=np.array([1, 2, 3, 4], dtype=self.dtype))
        yarr = self.SimpleArray(
            array=np.array([5, 6, 7, 8], dtype=self.dtype))
        pp_arr_aligned = self.PointPad(
            x=xarr, y=yarr, clone=True, alignment=16)
        self.assertEqual(pp_arr_aligned.alignment, 16)
        self.assertEqual(pp_arr_aligned.ndim, 2)

        # Test invalid alignment values during array initialization
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(x=xarr, y=yarr, clone=True, alignment=12)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(x=xarr, y=yarr, clone=False, alignment=48)

        # Test alignment for 3D arrays
        zarr = self.SimpleArray(
            array=np.array([9, 10, 11, 12, 13, 14, 15, 16],
                           dtype=self.dtype))
        xarr2 = self.SimpleArray(
            array=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=self.dtype))
        yarr2 = self.SimpleArray(
            array=np.array([9, 10, 11, 12, 13, 14, 15, 16],
                           dtype=self.dtype))
        pp_arr_3d_aligned = self.PointPad(
            x=xarr2, y=yarr2, z=zarr, clone=True, alignment=32)
        self.assertEqual(pp_arr_3d_aligned.alignment, 32)
        self.assertEqual(pp_arr_3d_aligned.ndim, 3)

        # Test invalid alignment values for 3D arrays
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(
                x=xarr2, y=yarr2, z=zarr, clone=True, alignment=100)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(
                x=xarr2, y=yarr2, z=zarr, clone=False, alignment=512)

    def test_alignment_preservation(self):
        # Test that alignment is preserved during operations
        # like append and expand
        element_size = 4 if self.dtype == 'float32' else 8

        # Test alignment preservation for 2D PointPad
        nelem_32 = 32 // element_size
        pp2d_32 = self.PointPad(ndim=2, nelem=nelem_32, alignment=32)
        self.assertEqual(pp2d_32.alignment, 32)

        pp2d_32.append(1.0, 2.0)
        pp2d_32.append(3.0, 4.0)
        self.assertEqual(pp2d_32.alignment, 32)

        pp2d_32.expand(64)
        self.assertEqual(pp2d_32.alignment, 32)

        packed = pp2d_32.pack_array()
        self.assertEqual(packed.nbytes % 32, 0)

        # Test alignment preservation for 3D PointPad
        nelem_64 = 64 // element_size
        pp3d_64 = self.PointPad(ndim=3, nelem=nelem_64, alignment=64)
        self.assertEqual(pp3d_64.alignment, 64)

        pp3d_64.append(1.0, 2.0, 3.0)
        pp3d_64.append(4.0, 5.0, 6.0)
        self.assertEqual(pp3d_64.alignment, 64)

        pp3d_64.expand(128)
        self.assertEqual(pp3d_64.alignment, 64)

        packed_3d = pp3d_64.pack_array()
        self.assertEqual(packed_3d.nbytes % 64, 0)

        # Test alignment preservation for exceeding initial size
        nelem_16 = 16 // element_size
        pp2d_16 = self.PointPad(ndim=2, nelem=nelem_16, alignment=16)
        for i in range(10):
            pp2d_16.append(float(i), float(i + 1))
        self.assertEqual(pp2d_16.alignment, 16)

        pp2d_16.expand(200)
        self.assertEqual(pp2d_16.alignment, 16)

        packed_2d_16 = pp2d_16.pack_array()
        self.assertEqual(packed_2d_16.nbytes % 16, 0)

        # Test unaligned PointPad
        pp2d_unaligned = self.PointPad(ndim=2, nelem=5)
        self.assertEqual(pp2d_unaligned.alignment, 0)

        pp2d_unaligned.append(1.0, 2.0)
        self.assertEqual(pp2d_unaligned.alignment, 0)

        pp2d_unaligned.expand(20)
        self.assertEqual(pp2d_unaligned.alignment, 0)

    def test_alignment_valid_values(self):
        # Test valid alignment values (0, 16, 32, 64) for PointPad
        element_size = 4 if self.dtype == 'float32' else 8

        for alignment in [0, 16, 32, 64]:
            if alignment == 0:
                nelem = 4
            else:
                nelem = alignment // element_size

            pp2d = self.PointPad(
                ndim=2, nelem=nelem, alignment=alignment)
            self.assertEqual(pp2d.alignment, alignment)

            nelem_3d = max(nelem, 8)
            pp3d = self.PointPad(
                ndim=3, nelem=nelem_3d, alignment=alignment)
            self.assertEqual(pp3d.alignment, alignment)

            if alignment == 0:
                arr_nelem = 3
            else:
                arr_nelem = max(alignment // element_size, 2)

            xarr = self.SimpleArray(
                array=np.arange(arr_nelem, dtype=self.dtype))
            yarr = self.SimpleArray(
                array=np.arange(arr_nelem, arr_nelem * 2,
                                dtype=self.dtype))
            pp_arr = self.PointPad(
                x=xarr, y=yarr, clone=True, alignment=alignment)
            self.assertEqual(pp_arr.alignment, alignment)

            zarr = self.SimpleArray(
                array=np.arange(arr_nelem * 2, arr_nelem * 3,
                                dtype=self.dtype))
            pp_arr_3d = self.PointPad(
                x=xarr, y=yarr, z=zarr, clone=True, alignment=alignment)
            self.assertEqual(pp_arr_3d.alignment, alignment)

    def test_alignment_size_validation(self):
        # Test that alignment size validation works correctly
        element_size = 4 if self.dtype == 'float32' else 8
        test_cases = [
            (4, [
                (2, 5, 16,
                 f"BufferExpander::allocate: size {4 * 5} must be a multiple of alignment 16"),  # noqa E501
                (2, 7, 32,
                 f"BufferExpander::allocate: size {4 * 7} must be a multiple of alignment 32"),  # noqa E501
                (3, 9, 64,
                 f"BufferExpander::allocate: size {4 * 9} must be a multiple of alignment 64"),  # noqa E501
                (2, 11, 16,
                 f"BufferExpander::allocate: size {4 * 11} must be a multiple of alignment 16"),  # noqa E501
                (3, 13, 32,
                 f"BufferExpander::allocate: size {4 * 13} must be a multiple of alignment 32"),  # noqa E501
            ]),
            (8, [
                (2, 3, 32,
                 f"BufferExpander::allocate: size {8 * 3} must be a multiple of alignment 32"),  # noqa E501
                (2, 5, 64,
                 f"BufferExpander::allocate: size {8 * 5} must be a multiple of alignment 64"),  # noqa E501
                (3, 7, 64,
                 f"BufferExpander::allocate: size {8 * 7} must be a multiple of alignment 64"),  # noqa E501
                (2, 9, 16,
                 f"BufferExpander::allocate: size {8 * 9} must be a multiple of alignment 16"),  # noqa E501
                (3, 11, 32,
                 f"BufferExpander::allocate: size {8 * 11} must be a multiple of alignment 32"),  # noqa E501
            ]),
        ]

        for size, cases in test_cases:
            if element_size == size:
                for ndim, nelem, alignment, error_msg in cases:
                    with self.assertRaisesRegex(ValueError, error_msg):
                        self.PointPad(
                            ndim=ndim, nelem=nelem, alignment=alignment)


class PointPadFp32TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = modmesh.SimpleArrayFloat32
        self.Point = modmesh.Point3dFp32
        self.PointPad = modmesh.PointPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class PointPadFp64TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = modmesh.SimpleArrayFloat64
        self.Point = modmesh.Point3dFp64
        self.PointPad = modmesh.PointPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class SegmentPadTB(ModMeshTB):

    def test_ndim(self):
        sp2d = self.SegmentPad(ndim=2)
        self.assertEqual(sp2d.ndim, 2)
        sp3d = self.SegmentPad(ndim=3)
        self.assertEqual(sp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.SegmentPad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.SegmentPad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.SegmentPad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.SegmentPad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.SegmentPad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.SegmentPad(ndim=4, nelem=5)

    def test_construct_2d(self):
        x0arr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        x1arr = self.SimpleArray(array=np.array([-1, -2, -3],
                                                dtype=self.dtype))
        y1arr = self.SimpleArray(array=np.array([-4, -5, -6],
                                                dtype=self.dtype))
        sp = self.SegmentPad(x0=x0arr, y0=y0arr, x1=x1arr, y1=y1arr,
                             clone=False)
        self.assertEqual(sp.ndim, 2)
        self.assert_allclose(sp.x0, [1, 2, 3])
        self.assert_allclose(sp.y0, [4, 5, 6])
        self.assert_allclose(sp.x1, [-1, -2, -3])
        self.assert_allclose(sp.y1, [-4, -5, -6])
        self.assertEqual(len(sp.z0), 0)
        self.assertEqual(len(sp.z1), 0)

        # Test zero-copy writing
        sp.x0[1] = 200.2
        sp.y0[0] = -700.3
        sp.x1[1] = -200.2
        sp.y1[0] = 700.3
        self.assert_allclose(list(sp[0]), [[1, -700.3, 0], [-1, 700.3, 0]])
        self.assert_allclose(list(sp[1]), [[200.2, 5, 0], [-200.2, -5, 0]])
        self.assert_allclose(list(sp[2]), [[3, 6, 0], [-3, -6, 0]])

        sp2 = self.SegmentPad(ndim=2, nelem=3)
        for i in range(len(sp)):
            sp2.set_at(i, sp.get_at(i).x0, sp.get_at(i).y0,
                       sp.get_at(i).x1, sp.get_at(i).y1)
        self.assert_allclose(sp2.x0, [1, 200.2, 3])
        self.assert_allclose(sp2.y0, [-700.3, 5, 6])
        self.assert_allclose(sp2.x1, [-1, -200.2, -3])
        self.assert_allclose(sp2.y1, [700.3, -5, -6])
        self.assertEqual(len(sp2.z0), 0)
        self.assertEqual(len(sp2.z1), 0)

        packed = sp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 4))
        self.assert_allclose(list(packed[0]), (1, -700.3, -1, 700.3))
        self.assert_allclose(list(packed[1]), (200.2, 5, -200.2, -5))
        self.assert_allclose(list(packed[2]), (3, 6, -3, -6))

    def test_construct_3d(self):
        Point = self.Point

        x0arr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.SimpleArray(array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.SimpleArray(array=np.array([-1, -2, -3],
                                                dtype=self.dtype))
        y1arr = self.SimpleArray(array=np.array([-4, -5, -6],
                                                dtype=self.dtype))
        z1arr = self.SimpleArray(array=np.array([-7, -8, -9],
                                                dtype=self.dtype))
        sp = self.SegmentPad(x0=x0arr, y0=y0arr, z0=z0arr,
                             x1=x1arr, y1=y1arr, z1=z1arr, clone=False)
        self.assertEqual(sp.ndim, 3)
        self.assert_allclose(sp.x0, [1, 2, 3])
        self.assert_allclose(sp.y0, [4, 5, 6])
        self.assert_allclose(sp.z0, [7, 8, 9])
        self.assert_allclose(sp.x1, [-1, -2, -3])
        self.assert_allclose(sp.y1, [-4, -5, -6])
        self.assert_allclose(sp.z1, [-7, -8, -9])

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.x0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.y1_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.z0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.p0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.get_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.set_at(3, self.Segment(Point(0, 0, 0), Point(0, 0, 0)))

        # Test zero-copy writing
        sp.x0[1] = 200.2
        sp.y0[0] = -700.3
        sp.z0[2] = 213.9
        sp.x1[1] = -200.2
        sp.y1[0] = 700.3
        sp.z1[2] = -213.9
        self.assert_allclose(list(sp[0]), [[1, -700.3, 7], [-1, 700.3, -7]])
        self.assert_allclose(list(sp[1]), [[200.2, 5, 8], [-200.2, -5, -8]])
        self.assert_allclose(list(sp[2]), [[3, 6, 213.9], [-3, -6, -213.9]])

        sp2 = self.SegmentPad(ndim=3, nelem=3)
        for i in range(len(sp)):
            sp2.set_at(i, sp.get_at(i).x0, sp.get_at(i).y0, sp.get_at(i).z0,
                       sp.get_at(i).x1, sp.get_at(i).y1, sp.get_at(i).z1)
        self.assert_allclose(sp2.x0, [1, 200.2, 3])
        self.assert_allclose(sp2.y0, [-700.3, 5, 6])
        self.assert_allclose(sp2.z0, [7, 8, 213.9])
        self.assert_allclose(sp2.x1, [-1, -200.2, -3])
        self.assert_allclose(sp2.y1, [700.3, -5, -6])
        self.assert_allclose(sp2.z1, [-7, -8, -213.9])

        packed = sp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 6))
        self.assert_allclose(list(packed[0]), (1, -700.3, 7, -1, 700.3, -7))
        self.assert_allclose(list(packed[1]), (200.2, 5, 8, -200.2, -5, -8))
        self.assert_allclose(list(packed[2]), (3, 6, 213.9, -3, -6, -213.9))

    def test_append_2d(self):
        Point = self.Point
        Segment = self.Segment

        sp = self.SegmentPad(ndim=2)
        self.assertEqual(sp.ndim, 2)
        self.assertEqual(len(sp), 0)
        sp.append(Segment(Point(1.1, 2.2, 0.0), Point(7.1, 8.2, 0.0)))
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        sp.append(Point(1.1 * 3, 2.2 * 3), Point(7.1 * 3, 8.2 * 3))
        self.assertEqual(len(sp), 2)
        self.assert_allclose(sp.x0_at(1), 1.1 * 3)
        self.assert_allclose(sp.y0_at(1), 2.2 * 3)
        self.assert_allclose(sp.x1_at(1), 7.1 * 3)
        self.assert_allclose(sp.y1_at(1), 8.2 * 3)
        sp.append(1.1 * 3.1, 2.2 * 3.1, 7.1 * 3.1, 8.2 * 3.1)
        self.assertEqual(len(sp), 3)
        self.assert_allclose(sp.x0_at(2), 1.1 * 3.1)
        self.assert_allclose(sp.y0_at(2), 2.2 * 3.1)
        self.assert_allclose(sp.x1_at(2), 7.1 * 3.1)
        self.assert_allclose(sp.y1_at(2), 8.2 * 3.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 3 but is 2"):
            sp.append(3.2, 4.1, 5.7, 3.2, 4.1, 5.7)
        self.assertEqual(len(sp), 3)

        # Test batch interface
        self.assert_allclose(sp.x0, [1.1, 1.1 * 3, 1.1 * 3.1])
        self.assert_allclose(sp.y0, [2.2, 2.2 * 3, 2.2 * 3.1])
        self.assert_allclose(sp.x1, [7.1, 7.1 * 3, 7.1 * 3.1])
        self.assert_allclose(sp.y1, [8.2, 8.2 * 3, 8.2 * 3.1])
        sp.x0[0] = -10.9
        sp.x0.ndarray[2] = -13.2
        sp.x1[0] = 10.9
        sp.x1.ndarray[2] = 13.2
        self.assert_allclose(sp.x0_at(0), -10.9)
        self.assert_allclose(sp.x0_at(1), 1.1 * 3)
        self.assert_allclose(sp.x0_at(2), -13.2)
        self.assert_allclose(sp.x1_at(0), 10.9)
        self.assert_allclose(sp.x1_at(1), 7.1 * 3)
        self.assert_allclose(sp.x1_at(2), 13.2)
        sp.y0[1] = -0.93
        sp.y0.ndarray[2] = 29.1
        sp.y1[1] = 0.93
        sp.y1.ndarray[2] = -29.1
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.y0_at(1), -0.93)
        self.assert_allclose(sp.y0_at(2), 29.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.y1_at(1), 0.93)
        self.assert_allclose(sp.y1_at(2), -29.1)
        self.assertEqual(len(sp.z0), 0)
        self.assertEqual(len(sp.z1), 0)

        nseg = len(sp)
        sp.extend_with(sp)
        for i in range(nseg):
            self.assertEqual(sp[i], sp[nseg + i])

    def test_append_3d(self):
        Point = self.Point
        Segment = self.Segment

        sp = self.SegmentPad(ndim=3)
        self.assertEqual(sp.ndim, 3)
        self.assertEqual(len(sp), 0)
        sp.append(s=Segment(Point(1.1, 2.2, 3.3), Point(7.1, 8.2, 9.3)))
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.z0_at(0), 3.3)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.z1_at(0), 9.3)
        sp.append(p0=Point(1.1 * 5, 2.2 * 5, 3.3 * 5),
                  p1=Point(7.1 * 5, 8.2 * 5, 9.3 * 5))
        self.assertEqual(len(sp), 2)
        self.assert_allclose(sp.x0_at(1), 1.1 * 5)
        self.assert_allclose(sp.y0_at(1), 2.2 * 5)
        self.assert_allclose(sp.z0_at(1), 3.3 * 5)
        self.assert_allclose(sp.x1_at(1), 7.1 * 5)
        self.assert_allclose(sp.y1_at(1), 8.2 * 5)
        self.assert_allclose(sp.z1_at(1), 9.3 * 5)
        sp.append(x0=1.1 * 5.1, y0=2.2 * 5.1, z0=3.3 * 5.1,
                  x1=7.1 * 5.1, y1=8.2 * 5.1, z1=9.3 * 5.1)
        self.assertEqual(len(sp), 3)
        self.assert_allclose(sp.x0_at(2), 1.1 * 5.1)
        self.assert_allclose(sp.y0_at(2), 2.2 * 5.1)
        self.assert_allclose(sp.z0_at(2), 3.3 * 5.1)
        self.assert_allclose(sp.x1_at(2), 7.1 * 5.1)
        self.assert_allclose(sp.y1_at(2), 8.2 * 5.1)
        self.assert_allclose(sp.z1_at(2), 9.3 * 5.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 2 but is 3"):
            sp.append(3.2, 4.1, 5.2, 6.2)
        self.assertEqual(len(sp), 3)

        # Test batch interface
        self.assert_allclose(sp.x0, [1.1, 1.1 * 5, 1.1 * 5.1])
        self.assert_allclose(sp.y0, [2.2, 2.2 * 5, 2.2 * 5.1])
        self.assert_allclose(sp.z0, [3.3, 3.3 * 5, 3.3 * 5.1])
        self.assert_allclose(sp.x1, [7.1, 7.1 * 5, 7.1 * 5.1])
        self.assert_allclose(sp.y1, [8.2, 8.2 * 5, 8.2 * 5.1])
        self.assert_allclose(sp.z1, [9.3, 9.3 * 5, 9.3 * 5.1])
        sp.x0[0] = -10.9
        sp.x0.ndarray[2] = -13.2
        sp.x1[0] = 10.9
        sp.x1.ndarray[2] = 13.2
        self.assert_allclose(sp.x0_at(0), -10.9)
        self.assert_allclose(sp.x0_at(1), 1.1 * 5)
        self.assert_allclose(sp.x0_at(2), -13.2)
        self.assert_allclose(sp.x1_at(0), 10.9)
        self.assert_allclose(sp.x1_at(1), 7.1 * 5)
        self.assert_allclose(sp.x1_at(2), 13.2)
        sp.y0[1] = -0.93
        sp.y0.ndarray[2] = 29.1
        sp.y1[1] = 0.93
        sp.y1.ndarray[2] = -29.1
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.y0_at(1), -0.93)
        self.assert_allclose(sp.y0_at(2), 29.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.y1_at(1), 0.93)
        self.assert_allclose(sp.y1_at(2), -29.1)
        sp.z0[0] = 2.31
        sp.z0.ndarray[1] = 8.23
        sp.z1[0] = -2.31
        sp.z1.ndarray[1] = -8.23
        self.assert_allclose(sp.z0_at(0), 2.31)
        self.assert_allclose(sp.z0_at(1), 8.23)
        self.assert_allclose(sp.z0_at(2), 3.3 * 5.1)
        self.assert_allclose(sp.z1_at(0), -2.31)
        self.assert_allclose(sp.z1_at(1), -8.23)
        self.assert_allclose(sp.z1_at(2), 9.3 * 5.1)

        nseg = len(sp)
        sp.extend_with(sp)
        for i in range(nseg):
            self.assertEqual(sp[i], sp[nseg + i])

        # Assert the equality between value array and PointPad
        self.assert_allclose(list(sp.x0), list(sp.p0.x))
        self.assert_allclose(list(sp.y0), list(sp.p0.y))
        self.assert_allclose(list(sp.z0), list(sp.p0.z))
        self.assert_allclose(list(sp.x1), list(sp.p1.x))
        self.assert_allclose(list(sp.y1), list(sp.p1.y))
        self.assert_allclose(list(sp.z1), list(sp.p1.z))

    def test_mirror_2d(self):
        SegmentPad = self.SegmentPad

        sp = SegmentPad(ndim=2)
        sp.append(1.0, 2.0, 3.0, 4.0)
        sp.append(5.0, 6.0, 7.0, 8.0)

        sp.mirror('x')
        self.assert_allclose(sp.x0_at(0), 1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.x1_at(0), 3.0)
        self.assert_allclose(sp.y1_at(0), -4.0)
        self.assert_allclose(sp.x0_at(1), 5.0)
        self.assert_allclose(sp.y0_at(1), -6.0)
        self.assert_allclose(sp.x1_at(1), 7.0)
        self.assert_allclose(sp.y1_at(1), -8.0)

        sp.mirror('y')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.x1_at(0), -3.0)
        self.assert_allclose(sp.y1_at(0), -4.0)
        self.assert_allclose(sp.x0_at(1), -5.0)
        self.assert_allclose(sp.y0_at(1), -6.0)
        self.assert_allclose(sp.x1_at(1), -7.0)
        self.assert_allclose(sp.y1_at(1), -8.0)

    def test_mirror_3d(self):
        SegmentPad = self.SegmentPad

        sp = SegmentPad(ndim=3)
        sp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        sp.append(7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        sp.mirror('z')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.z0_at(0), 3.0)
        self.assert_allclose(sp.x1_at(0), -4.0)
        self.assert_allclose(sp.y1_at(0), -5.0)
        self.assert_allclose(sp.z1_at(0), 6.0)
        self.assert_allclose(sp.x0_at(1), -7.0)
        self.assert_allclose(sp.y0_at(1), -8.0)
        self.assert_allclose(sp.z0_at(1), 9.0)
        self.assert_allclose(sp.x1_at(1), -10.0)
        self.assert_allclose(sp.y1_at(1), -11.0)
        self.assert_allclose(sp.z1_at(1), 12.0)

        sp.mirror('X')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), 2.0)
        self.assert_allclose(sp.z0_at(0), -3.0)
        self.assert_allclose(sp.x1_at(0), -4.0)
        self.assert_allclose(sp.y1_at(0), 5.0)
        self.assert_allclose(sp.z1_at(0), -6.0)

        with self.assertRaisesRegex(
                ValueError,
                "SegmentPad::mirror: axis must be 'x', 'y', or 'z'"):
            sp.mirror('w')


class SegmentPadFp32TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = modmesh.SimpleArrayFloat32
        self.Point = modmesh.Point3dFp32
        self.PointPad = modmesh.PointPadFp32
        self.Segment = modmesh.Segment3dFp32
        self.SegmentPad = modmesh.SegmentPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class SegmentPadFp64TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = modmesh.SimpleArrayFloat64
        self.Point = modmesh.Point3dFp64
        self.PointPad = modmesh.PointPadFp64
        self.Segment = modmesh.Segment3dFp64
        self.SegmentPad = modmesh.SegmentPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class TrianglePadTB(ModMeshTB):

    def test_ndim(self):
        tp2d = self.TrianglePad(ndim=2)
        self.assertEqual(tp2d.ndim, 2)
        tp3d = self.TrianglePad(ndim=3)
        self.assertEqual(tp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.TrianglePad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.TrianglePad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.TrianglePad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.TrianglePad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.TrianglePad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.TrianglePad(ndim=4, nelem=5)

    def test_construct_2d(self):
        x0arr = self.SimpleArray(
            array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(
            array=np.array([4, 5, 6], dtype=self.dtype))
        x1arr = self.SimpleArray(
            array=np.array([7, 8, 9], dtype=self.dtype))
        y1arr = self.SimpleArray(
            array=np.array([10, 11, 12], dtype=self.dtype))
        x2arr = self.SimpleArray(
            array=np.array([13, 14, 15], dtype=self.dtype))
        y2arr = self.SimpleArray(
            array=np.array([16, 17, 18], dtype=self.dtype))
        tp = self.TrianglePad(x0=x0arr, y0=y0arr, x1=x1arr, y1=y1arr,
                              x2=x2arr, y2=y2arr, clone=False)
        self.assertEqual(tp.ndim, 2)
        self.assert_allclose(tp.x0, [1, 2, 3])
        self.assert_allclose(tp.y0, [4, 5, 6])
        self.assert_allclose(tp.x1, [7, 8, 9])
        self.assert_allclose(tp.y1, [10, 11, 12])
        self.assert_allclose(tp.x2, [13, 14, 15])
        self.assert_allclose(tp.y2, [16, 17, 18])
        self.assertEqual(len(tp.z0), 0)
        self.assertEqual(len(tp.z1), 0)
        self.assertEqual(len(tp.z2), 0)

        tp.y0[0] = -700.3
        tp.y1[0] = 900.5
        tp.y2[0] = 1100.7
        tp.x0[1] = 200.2
        tp.x1[1] = -800.4
        tp.x2[1] = -1000.6
        self.assert_allclose(
            list(tp[0]), [[1, -700.3, 0], [7, 900.5, 0], [13, 1100.7, 0]])
        self.assert_allclose(
            list(tp[1]),
            [[200.2, 5, 0], [-800.4, 11, 0], [-1000.6, 17, 0]])
        self.assert_allclose(
            list(tp[2]), [[3, 6, 0], [9, 12, 0], [15, 18, 0]])  # no change

        tp2 = self.TrianglePad(ndim=2, nelem=3)
        for i in range(len(tp)):
            tp2.set_at(i, tp.get_at(i).x0, tp.get_at(i).y0,
                       tp.get_at(i).x1, tp.get_at(i).y1,
                       tp.get_at(i).x2, tp.get_at(i).y2)
        self.assert_allclose(tp2.x0, [1, 200.2, 3])
        self.assert_allclose(tp2.y0, [-700.3, 5, 6])
        self.assert_allclose(tp2.x1, [7, -800.4, 9])
        self.assert_allclose(tp2.y1, [900.5, 11, 12])
        self.assert_allclose(tp2.x2, [13, -1000.6, 15])
        self.assert_allclose(tp2.y2, [1100.7, 17, 18])
        self.assertEqual(len(tp2.z0), 0)
        self.assertEqual(len(tp2.z1), 0)
        self.assertEqual(len(tp2.z2), 0)

        packed = tp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 6))
        self.assert_allclose(
            list(packed[0]), (1, -700.3, 7, 900.5, 13, 1100.7))
        self.assert_allclose(
            list(packed[1]), (200.2, 5, -800.4, 11, -1000.6, 17))
        self.assert_allclose(
            list(packed[2]), (3, 6, 9, 12, 15, 18))

    def test_construct_3d(self):
        x0arr = self.SimpleArray(
            array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(
            array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.SimpleArray(
            array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.SimpleArray(
            array=np.array([10, 11, 12], dtype=self.dtype))
        y1arr = self.SimpleArray(
            array=np.array([13, 14, 15], dtype=self.dtype))
        z1arr = self.SimpleArray(
            array=np.array([16, 17, 18], dtype=self.dtype))
        x2arr = self.SimpleArray(
            array=np.array([19, 20, 21], dtype=self.dtype))
        y2arr = self.SimpleArray(
            array=np.array([22, 23, 24], dtype=self.dtype))
        z2arr = self.SimpleArray(
            array=np.array([25, 26, 27], dtype=self.dtype))
        tp = self.TrianglePad(
            x0=x0arr, y0=y0arr, z0=z0arr,
            x1=x1arr, y1=y1arr, z1=z1arr,
            x2=x2arr, y2=y2arr, z2=z2arr,
            clone=False)
        self.assertEqual(tp.ndim, 3)
        self.assert_allclose(tp.x0, [1, 2, 3])
        self.assert_allclose(tp.y0, [4, 5, 6])
        self.assert_allclose(tp.z0, [7, 8, 9])
        self.assert_allclose(tp.x1, [10, 11, 12])
        self.assert_allclose(tp.y1, [13, 14, 15])
        self.assert_allclose(tp.z1, [16, 17, 18])
        self.assert_allclose(tp.x2, [19, 20, 21])
        self.assert_allclose(tp.y2, [22, 23, 24])
        self.assert_allclose(tp.z2, [25, 26, 27])

        tp.y0[0] = -300.3
        tp.y1[0] = 600.6
        tp.y2[0] = -900.9
        tp.x0[1] = 200.2
        tp.x1[1] = -500.5
        tp.x2[1] = 800.8
        tp.z0[2] = 400.4
        tp.z1[2] = -700.7
        tp.z2[2] = 1000.1
        self.assert_allclose(
            list(tp[0]),
            [[1, -300.3, 7], [10, 600.6, 16], [19, -900.9, 25]])  # y changed
        self.assert_allclose(
            list(tp[1]),
            [[200.2, 5, 8], [-500.5, 14, 17], [800.8, 23, 26]])  # x changed
        self.assert_allclose(
            list(tp[2]),
            [[3, 6, 400.4], [12, 15, -700.7], [21, 24, 1000.1]])  # z changed

        tp2 = self.TrianglePad(ndim=3, nelem=3)
        for i in range(len(tp)):
            tp2.set_at(
                i,
                tp.get_at(i).x0, tp.get_at(i).y0, tp.get_at(i).z0,
                tp.get_at(i).x1, tp.get_at(i).y1, tp.get_at(i).z1,
                tp.get_at(i).x2, tp.get_at(i).y2, tp.get_at(i).z2)
        self.assert_allclose(tp2.x0, [1, 200.2, 3])
        self.assert_allclose(tp2.y0, [-300.3, 5, 6])
        self.assert_allclose(tp2.z0, [7, 8, 400.4])
        self.assert_allclose(tp2.x1, [10, -500.5, 12])
        self.assert_allclose(tp2.y1, [600.6, 14, 15])
        self.assert_allclose(tp2.z1, [16, 17, -700.7])
        self.assert_allclose(tp2.x2, [19, 800.8, 21])
        self.assert_allclose(tp2.y2, [-900.9, 23, 24])
        self.assert_allclose(tp2.z2, [25, 26, 1000.1])

        packed = tp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 9))
        self.assert_allclose(
            list(packed[0]),
            (1, -300.3, 7, 10, 600.6, 16, 19, -900.9, 25))
        self.assert_allclose(
            list(packed[1]),
            (200.2, 5, 8, -500.5, 14, 17, 800.8, 23, 26))
        self.assert_allclose(
            list(packed[2]),
            (3, 6, 400.4, 12, 15, -700.7, 21, 24, 1000.1))

    def test_append_2d(self):
        tp = self.TrianglePad(ndim=2)
        self.assertEqual(tp.ndim, 2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        tp.append(
            self.Point(7.0, 8.0),
            self.Point(9.0, 10.0),
            self.Point(11.0, 12.0))
        tp.append(self.Triangle(
            self.Point(13.0, 14.0),
            self.Point(15.0, 16.0),
            self.Point(17.0, 18.0)))

        self.assertEqual(tp.x0_at(0), 1.0)
        self.assertEqual(tp.y0_at(0), 2.0)
        self.assertEqual(tp.x1_at(0), 3.0)
        self.assertEqual(tp.y1_at(0), 4.0)
        self.assertEqual(tp.x2_at(0), 5.0)
        self.assertEqual(tp.y2_at(0), 6.0)

        self.assertEqual(tp.x0_at(1), 7.0)
        self.assertEqual(tp.y0_at(1), 8.0)
        self.assertEqual(tp.x1_at(1), 9.0)
        self.assertEqual(tp.y1_at(1), 10.0)
        self.assertEqual(tp.x2_at(1), 11.0)
        self.assertEqual(tp.y2_at(1), 12.0)

        self.assertEqual(tp.x0_at(2), 13.0)
        self.assertEqual(tp.y0_at(2), 14.0)
        self.assertEqual(tp.x1_at(2), 15.0)
        self.assertEqual(tp.y1_at(2), 16.0)
        self.assertEqual(tp.x2_at(2), 17.0)
        self.assertEqual(tp.y2_at(2), 18.0)

    def test_append_3d(self):
        tp = self.TrianglePad(ndim=3)
        self.assertEqual(tp.ndim, 3)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        tp.append(
            self.Point(10.0, 11.0, 12.0),
            self.Point(13.0, 14.0, 15.0),
            self.Point(16.0, 17.0, 18.0))
        tp.append(self.Triangle(
            self.Point(19.0, 20.0, 21.0),
            self.Point(22.0, 23.0, 24.0),
            self.Point(25.0, 26.0, 27.0)))

        self.assertEqual(tp.x0_at(0), 1.0)
        self.assertEqual(tp.y0_at(0), 2.0)
        self.assertEqual(tp.z0_at(0), 3.0)
        self.assertEqual(tp.x1_at(0), 4.0)
        self.assertEqual(tp.y1_at(0), 5.0)
        self.assertEqual(tp.z1_at(0), 6.0)
        self.assertEqual(tp.x2_at(0), 7.0)
        self.assertEqual(tp.y2_at(0), 8.0)
        self.assertEqual(tp.z2_at(0), 9.0)

        self.assertEqual(tp.x0_at(1), 10.0)
        self.assertEqual(tp.y0_at(1), 11.0)
        self.assertEqual(tp.z0_at(1), 12.0)
        self.assertEqual(tp.x1_at(1), 13.0)
        self.assertEqual(tp.y1_at(1), 14.0)
        self.assertEqual(tp.z1_at(1), 15.0)
        self.assertEqual(tp.x2_at(1), 16.0)
        self.assertEqual(tp.y2_at(1), 17.0)
        self.assertEqual(tp.z2_at(1), 18.0)

        self.assertEqual(tp.x0_at(2), 19.0)
        self.assertEqual(tp.y0_at(2), 20.0)
        self.assertEqual(tp.z0_at(2), 21.0)
        self.assertEqual(tp.x1_at(2), 22.0)
        self.assertEqual(tp.y1_at(2), 23.0)
        self.assertEqual(tp.z1_at(2), 24.0)
        self.assertEqual(tp.x2_at(2), 25.0)
        self.assertEqual(tp.y2_at(2), 26.0)
        self.assertEqual(tp.z2_at(2), 27.0)

    def test_mirror_2d_x(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        tp.mirror('x')

        self.assert_allclose(tp.x0_at(0), 1.0)
        self.assert_allclose(tp.y0_at(0), -2.0)
        self.assert_allclose(tp.x1_at(0), 3.0)
        self.assert_allclose(tp.y1_at(0), -4.0)
        self.assert_allclose(tp.x2_at(0), 5.0)
        self.assert_allclose(tp.y2_at(0), -6.0)

    def test_mirror_2d_y(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        tp.mirror('y')

        self.assert_allclose(tp.x0_at(0), -1.0)
        self.assert_allclose(tp.y0_at(0), 2.0)
        self.assert_allclose(tp.x1_at(0), -3.0)
        self.assert_allclose(tp.y1_at(0), 4.0)
        self.assert_allclose(tp.x2_at(0), -5.0)
        self.assert_allclose(tp.y2_at(0), 6.0)

    def test_mirror_3d_z(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=3)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        tp.mirror('z')

        self.assert_allclose(tp.x0_at(0), -1.0)
        self.assert_allclose(tp.y0_at(0), -2.0)
        self.assert_allclose(tp.z0_at(0), 3.0)
        self.assert_allclose(tp.x1_at(0), -4.0)
        self.assert_allclose(tp.y1_at(0), -5.0)
        self.assert_allclose(tp.z1_at(0), 6.0)
        self.assert_allclose(tp.x2_at(0), -7.0)
        self.assert_allclose(tp.y2_at(0), -8.0)
        self.assert_allclose(tp.z2_at(0), 9.0)

        with self.assertRaisesRegex(
                ValueError,
                "TrianglePad::mirror: axis must be 'x', 'y', or 'z'"):
            tp.mirror('w')


class TrianglePadFp32TC(TrianglePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = modmesh.SimpleArrayFloat32
        self.Point = modmesh.Point3dFp32
        self.Triangle = modmesh.Triangle3dFp32
        self.TrianglePad = modmesh.TrianglePadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class TrianglePadFp64TC(TrianglePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = modmesh.SimpleArrayFloat64
        self.Point = modmesh.Point3dFp64
        self.Triangle = modmesh.Triangle3dFp64
        self.TrianglePad = modmesh.TrianglePadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class CurvePadTB(ModMeshTB):

    def test_ndim(self):
        cp2d = self.CurvePad(ndim=2)
        self.assertEqual(cp2d.ndim, 2)
        cp3d = self.CurvePad(ndim=3)
        self.assertEqual(cp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.CurvePad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.CurvePad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.CurvePad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.CurvePad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.CurvePad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.CurvePad(ndim=4, nelem=5)

    def test_append_2d(self):
        cp = self.CurvePad(ndim=2)
        self.assertEqual(cp.ndim, 2)
        self.assertEqual(len(cp), 0)

        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)

        self.assertEqual(cp.x0_at(0), 0)
        self.assertEqual(cp.y0_at(0), 0)
        self.assertEqual(cp.x1_at(0), 1)
        self.assertEqual(cp.y1_at(0), 1)
        self.assertEqual(cp.x2_at(0), 3)
        self.assertEqual(cp.y2_at(0), 1)
        self.assertEqual(cp.x3_at(0), 4)
        self.assertEqual(cp.y3_at(0), 0)

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z0_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z1_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z2_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z3_at(0)

        b = cp[0]
        self.assertEqual(len(b), 4)
        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])

        p0 = self.Point(7, 8, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        b = self.Bezier(p0, p1, p2, p3)
        cp[0] = b
        self.assertEqual(list(cp[0][0]), [7, 8, 0])
        self.assertEqual(list(cp[0][1]), [1, 1, 0])
        self.assertEqual(list(cp[0][2]), [3, 1, 0])
        self.assertEqual(list(cp[0][3]), [4, 0, 0])

    def test_append_3d(self):
        cp = self.CurvePad(ndim=3)
        self.assertEqual(cp.ndim, 3)
        self.assertEqual(len(cp), 0)

        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)

        self.assertEqual(cp.x0_at(0), 0)
        self.assertEqual(cp.y0_at(0), 0)
        self.assertEqual(cp.z0_at(0), 0)
        self.assertEqual(cp.x1_at(0), 1)
        self.assertEqual(cp.y1_at(0), 1)
        self.assertEqual(cp.z1_at(0), 0)
        self.assertEqual(cp.x2_at(0), 3)
        self.assertEqual(cp.y2_at(0), 1)
        self.assertEqual(cp.z2_at(0), 0)
        self.assertEqual(cp.x3_at(0), 4)
        self.assertEqual(cp.y3_at(0), 0)
        self.assertEqual(cp.z3_at(0), 0)

        p0 = self.Point(7, 8, -3)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        b = self.Bezier(p0, p1, p2, p3)
        cp[0] = b
        self.assertEqual(list(cp[0][0]), [7, 8, -3])
        self.assertEqual(list(cp[0][1]), [1, 1, 0])
        self.assertEqual(list(cp[0][2]), [3, 1, 0])
        self.assertEqual(list(cp[0][3]), [4, 0, 0])

        b2 = self.Bezier(
            p0=self.Point(0, 0, 1),
            p1=self.Point(1.3, 1.921, 2),
            p2=self.Point(3.2, 1.224, 3),
            p3=self.Point(4.87, 0.12, 4))
        cp.append(c=b2)
        self.assertEqual(len(cp), 2)
        # Assert the equality between value array and PointPad
        self.assert_allclose(list(cp.x0), list(cp.p0.x))
        self.assert_allclose(list(cp.y0), list(cp.p0.y))
        self.assert_allclose(list(cp.z0), list(cp.p0.z))
        self.assert_allclose(list(cp.x1), list(cp.p1.x))
        self.assert_allclose(list(cp.y1), list(cp.p1.y))
        self.assert_allclose(list(cp.z1), list(cp.p1.z))
        self.assert_allclose(list(cp.x2), list(cp.p2.x))
        self.assert_allclose(list(cp.y2), list(cp.p2.y))
        self.assert_allclose(list(cp.z2), list(cp.p2.z))
        self.assert_allclose(list(cp.x3), list(cp.p3.x))
        self.assert_allclose(list(cp.y3), list(cp.p3.y))
        self.assert_allclose(list(cp.z3), list(cp.p3.z))
        # Check the value
        self.assert_allclose(list(cp.x0), [7, 0])
        self.assert_allclose(list(cp.y0), [8, 0])
        self.assert_allclose(list(cp.z0), [-3, 1])
        self.assert_allclose(list(cp.x1), [1, 1.3])
        self.assert_allclose(list(cp.y1), [1, 1.921])
        self.assert_allclose(list(cp.z1), [0, 2])
        self.assert_allclose(list(cp.x2), [3, 3.2])
        self.assert_allclose(list(cp.y2), [1, 1.224])
        self.assert_allclose(list(cp.z2), [0, 3])
        self.assert_allclose(list(cp.x3), [4, 4.87])
        self.assert_allclose(list(cp.y3), [0, 0.12])
        self.assert_allclose(list(cp.z3), [0, 4])

    def test_sample_2d(self):
        CurvePad = self.CurvePad
        Point = self.Point
        Bezier = self.Bezier

        cp = CurvePad(ndim=3)
        p0 = Point(0, 0, 0)
        p1 = Point(1, 1, 0)
        p2 = Point(3, 1, 0)
        p3 = Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)
        p4 = Point(5, 0, 0)
        p5 = Point(5.5, 1, 0)
        p6 = Point(6.5, 1, 0)
        p7 = Point(7, 0, 0)
        c = Bezier(p0=p4, p1=p5, p2=p6, p3=p7)
        cp.append(c)
        self.assertEqual(len(cp), 2)

        # Sample to create segment pad
        sp = cp.sample(length=0.5)
        self.assertEqual(len(sp), 10)

        # The connectivity of the first curve
        self.assertEqual(p0, sp[0].p0)
        self.assertEqual(sp[0].p1, sp[1].p0)
        self.assertEqual(sp[1].p1, sp[2].p0)
        self.assertEqual(sp[2].p1, sp[3].p0)
        self.assertEqual(sp[3].p1, sp[4].p0)
        self.assertEqual(sp[4].p1, sp[5].p0)
        self.assertEqual(sp[5].p1, sp[6].p0)
        self.assertEqual(sp[6].p1, p3)

        # The connectivity of the second curve
        self.assertEqual(p4, sp[7].p0)
        self.assertEqual(sp[7].p1, sp[8].p0)
        self.assertEqual(sp[8].p1, sp[9].p0)
        self.assertEqual(sp[9].p1, p7)

        # Test for the segment coordinates of the first curve
        self.assert_allclose(list(sp[0].p0),
                             [0.0, 0.0, 0.0])
        self.assert_allclose(list(sp[0].p1),
                             [0.48396501457725954, 0.3673469387755103, 0.0])
        self.assert_allclose(list(sp[1].p0),
                             [0.48396501457725954, 0.3673469387755103, 0.0])
        self.assert_allclose(list(sp[1].p1),
                             [1.0553935860058308, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[2].p0),
                             [1.0553935860058308, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[2].p1),
                             [1.6793002915451893, 0.7346938775510203, 0.0])
        self.assert_allclose(list(sp[3].p0),
                             [1.6793002915451893, 0.7346938775510203, 0.0])
        self.assert_allclose(list(sp[3].p1),
                             [2.3206997084548107, 0.7346938775510206, 0.0])
        self.assert_allclose(list(sp[4].p0),
                             [2.3206997084548107, 0.7346938775510206, 0.0])
        self.assert_allclose(list(sp[4].p1),
                             [2.944606413994169, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[5].p0),
                             [2.944606413994169, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[5].p1),
                             [3.5160349854227406, 0.36734693877551033, 0.0])
        self.assert_allclose(list(sp[6].p0),
                             [3.5160349854227406, 0.36734693877551033, 0.0])
        self.assert_allclose(list(sp[6].p1),
                             [4.0, 0.0, 0.0])

        # Test for the segment coordinates of the second curve
        self.assert_allclose(list(sp[7].p0),
                             [5.0, 0.0, 0.0])
        self.assert_allclose(list(sp[7].p1),
                             [5.6296296296296315, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[8].p0),
                             [5.6296296296296315, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[8].p1),
                             [6.370370370370371, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[9].p0),
                             [6.370370370370371, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[9].p1),
                             [7.0, 0.0, 0.0])

    def test_mirror(self):
        CurvePad = self.CurvePad
        Point = self.Point

        cp = CurvePad(ndim=3)
        cp.append(Point(1, 2, 3), Point(4, 5, 6),
                  Point(7, 8, 9), Point(10, 11, 12))
        cp.append(Point(-1, -2, -3), Point(-4, -5, -6),
                  Point(-7, -8, -9), Point(-10, -11, -12))

        cp.mirror('x')
        self.assert_allclose(list(cp.x0), [1, -1])
        self.assert_allclose(list(cp.y0), [-2, 2])
        self.assert_allclose(list(cp.z0), [-3, 3])
        self.assert_allclose(list(cp.x1), [4, -4])
        self.assert_allclose(list(cp.y1), [-5, 5])
        self.assert_allclose(list(cp.z1), [-6, 6])
        self.assert_allclose(list(cp.x2), [7, -7])
        self.assert_allclose(list(cp.y2), [-8, 8])
        self.assert_allclose(list(cp.z2), [-9, 9])
        self.assert_allclose(list(cp.x3), [10, -10])
        self.assert_allclose(list(cp.y3), [-11, 11])
        self.assert_allclose(list(cp.z3), [-12, 12])

        cp.mirror('y')
        self.assert_allclose(list(cp.x0), [-1, 1])
        self.assert_allclose(list(cp.y0), [-2, 2])
        self.assert_allclose(list(cp.z0), [3, -3])
        self.assert_allclose(list(cp.x1), [-4, 4])
        self.assert_allclose(list(cp.y1), [-5, 5])
        self.assert_allclose(list(cp.z1), [6, -6])
        self.assert_allclose(list(cp.x2), [-7, 7])
        self.assert_allclose(list(cp.y2), [-8, 8])
        self.assert_allclose(list(cp.z2), [9, -9])
        self.assert_allclose(list(cp.x3), [-10, 10])
        self.assert_allclose(list(cp.y3), [-11, 11])
        self.assert_allclose(list(cp.z3), [12, -12])

        cp.mirror('Z')
        self.assert_allclose(list(cp.x0), [1, -1])
        self.assert_allclose(list(cp.y0), [2, -2])
        self.assert_allclose(list(cp.z0), [3, -3])
        self.assert_allclose(list(cp.x1), [4, -4])
        self.assert_allclose(list(cp.y1), [5, -5])
        self.assert_allclose(list(cp.z1), [6, -6])
        self.assert_allclose(list(cp.x2), [7, -7])
        self.assert_allclose(list(cp.y2), [8, -8])
        self.assert_allclose(list(cp.z2), [9, -9])
        self.assert_allclose(list(cp.x3), [10, -10])
        self.assert_allclose(list(cp.y3), [11, -11])
        self.assert_allclose(list(cp.z3), [12, -12])

        with self.assertRaisesRegex(
                ValueError, "CurvePad::mirror: axis must be 'x', 'y', or 'z'"):
            cp.mirror('w')


class CurvePadFp32TC(CurvePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = modmesh.SimpleArrayFloat32
        self.Point = modmesh.Point3dFp32
        self.PointPad = modmesh.PointPadFp32
        self.Segment = modmesh.Segment3dFp32
        self.SegmentPad = modmesh.SegmentPadFp32
        self.Bezier = modmesh.Bezier3dFp32
        self.CurvePad = modmesh.CurvePadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.5e-7
        return super().assert_allclose(*args, **kw)


class CurvePadFp64TC(CurvePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = modmesh.SimpleArrayFloat64
        self.Point = modmesh.Point3dFp64
        self.PointPad = modmesh.PointPadFp64
        self.Segment = modmesh.Segment3dFp64
        self.SegmentPad = modmesh.SegmentPadFp64
        self.Bezier = modmesh.Bezier3dFp64
        self.CurvePad = modmesh.CurvePadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class WorldTB(ModMeshTB):

    def test_bezier(self):
        Point = self.Point
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.nbezier, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 0 >= size 0"):
            w.bezier(0)

        # Add Bezier curve
        b = w.add_bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0),
                         p2=Point(3, 1, 0), p3=Point(4, 0, 0))
        self.assertEqual(w.nbezier, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 1 >= size 1"):
            w.bezier(1)
        b2 = w.add_bezier(b=self.Bezier(p0=Point(0, 0, 1), p1=Point(1, 1, 2),
                                        p2=Point(3, 1, 3), p3=Point(4, 0, 4)))
        self.assertEqual(w.nbezier, 2)
        w.bezier(1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 2 >= size 2"):
            w.bezier(2)

        # Check control points
        self.assertEqual(len(b), 4)
        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])
        self.assertEqual(len(b2), 4)
        self.assertEqual(list(b2[0]), [0, 0, 1])
        self.assertEqual(list(b2[1]), [1, 1, 2])
        self.assertEqual(list(b2[2]), [3, 1, 3])
        self.assertEqual(list(b2[3]), [4, 0, 4])

        # Check locus points
        segs = b.sample(nlocus=5)
        self.assertEqual(len(segs), 4)
        self.assert_allclose(
            list(segs[0]), [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[1]), [[0.90625, 0.5625, 0.0], [2.0, 0.75, 0.0]])
        self.assert_allclose(
            list(segs[2]), [[2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[3]), [[3.09375, 0.5625, 0.0], [4.0, 0.0, 0.0]])

    def test_beziers(self):
        cp = self.CurvePad(ndim=3)
        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)

        w = self.World()
        self.assertEqual(w.nbezier, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 0 >= size 0"):
            w.bezier(0)
        w.add_beziers(cp)
        self.assertEqual(w.nbezier, 1)
        b = w.bezier(0)

        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])

    def test_point(self):
        Point = self.Point
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.npoint, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 0 >= size 0"):
            w.point(0)

        # Add a point by object
        p = w.add_point(Point(0, 1, 2))
        self.assertEqual(list(p), [0, 1, 2])
        self.assertEqual(list(w.point(0)), [0, 1, 2])
        self.assertIsNot(p, w.point(0))
        self.assertEqual(w.npoint, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 1 >= size 1"):
            w.point(1)

        # Add a point by coordinate
        p = w.add_point(3.1415, 3.1416, 3.1417)
        self.assert_allclose(list(p), [3.1415, 3.1416, 3.1417])
        self.assert_allclose(list(w.point(1)), [3.1415, 3.1416, 3.1417])
        self.assertIsNot(p, w.point(1))
        self.assertEqual(w.npoint, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 2 >= size 2"):
            w.point(2)

        # Add many points
        for it in range(10):
            w.add_point(3.1415 + it, 3.1416 + it, 3.1417 + it)
            self.assertEqual(w.npoint, 2 + it + 1)

        # Array/batch interface
        pndarr = w.points.pack_array().ndarray
        self.assertEqual(pndarr.shape, (12, 3))
        self.assertEqual(w.npoint, 12)

    def test_segment(self):
        Point = self.Point
        Segment = self.Segment
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.nsegment, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 0 >= size 0"):
            w.segment(0)

        # Add a segment by object
        s = w.add_segment(s=Segment(Point(0, 1, 2), Point(7.1, 8.2, 9.3)))
        self.assert_allclose(list(s), [[0, 1, 2], [7.1, 8.2, 9.3]])
        self.assert_allclose(list(w.segment(0)), [[0, 1, 2], [7.1, 8.2, 9.3]])
        self.assertIsNot(s, w.segment(0))
        self.assertEqual(w.nsegment, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 1 >= size 1"):
            w.segment(1)

        # Add a segment by coordinate
        s = w.add_segment(Point(3.1415, 3.1416, 3.1417), Point(7.1, 8.2, 9.3))
        self.assert_allclose(list(s),
                             [[3.1415, 3.1416, 3.1417], [7.1, 8.2, 9.3]])
        self.assert_allclose(list(w.segment(1)),
                             [[3.1415, 3.1416, 3.1417], [7.1, 8.2, 9.3]])
        self.assertIsNot(s, w.segment(1))
        self.assertEqual(w.nsegment, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 2 >= size 2"):
            w.segment(2)

        # Add many segments
        for it in range(11):
            w.add_segment(Point(3.1415 + it, 3.1416 + it, 3.1417 + it),
                          Point(7.1 + it, 8.2 + it, 9.3 + it))
            self.assertEqual(w.nsegment, 2 + it + 1)

        # Array/batch interface
        sndarr = w.segments.pack_array().ndarray
        self.assertEqual(sndarr.shape, (13, 6))
        self.assertEqual(w.nsegment, 13)

    def test_segments(self):
        x0arr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.SimpleArray(array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.SimpleArray(array=np.array([-1, -2, -3],
                                                dtype=self.dtype))
        y1arr = self.SimpleArray(array=np.array([-4, -5, -6],
                                                dtype=self.dtype))
        z1arr = self.SimpleArray(array=np.array([-7, -8, -9],
                                                dtype=self.dtype))
        sp = self.SegmentPad(x0=x0arr, y0=y0arr, z0=z0arr,
                             x1=x1arr, y1=y1arr, z1=z1arr, clone=False)

        w = self.World()

        # Empty
        self.assertEqual(w.nsegment, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 0 >= size 0"):
            w.segment(0)
        # Add the SegmentPad
        w.add_segments(sp)
        self.assertEqual(w.nsegment, 3)

        s0 = w.segment(0)
        self.assert_allclose(list(s0[0]), [1, 4, 7])
        self.assert_allclose(list(s0[1]), [-1, -4, -7])
        s1 = w.segment(1)
        self.assert_allclose(list(s1[0]), [2, 5, 8])
        self.assert_allclose(list(s1[1]), [-2, -5, -8])
        s2 = w.segment(2)
        self.assert_allclose(list(s2[0]), [3, 6, 9])
        self.assert_allclose(list(s2[1]), [-3, -6, -9])


class WorldFp32TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = modmesh.SimpleArrayFloat32
        self.Point = modmesh.Point3dFp32
        self.Segment = modmesh.Segment3dFp32
        self.Bezier = modmesh.Bezier3dFp32
        self.SegmentPad = modmesh.SegmentPadFp32
        self.CurvePad = modmesh.CurvePadFp32
        self.World = modmesh.WorldFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp32, self.World)


class WorldFp64TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = modmesh.SimpleArrayFloat64
        self.Point = modmesh.Point3dFp64
        self.Segment = modmesh.Segment3dFp64
        self.Bezier = modmesh.Bezier3dFp64
        self.SegmentPad = modmesh.SegmentPadFp64
        self.CurvePad = modmesh.CurvePadFp64
        self.World = modmesh.WorldFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp64, self.World)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
