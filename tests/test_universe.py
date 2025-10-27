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
        self.assertEqual(list(p1), [-1, 2, 3])

        p2 = Point(1, 2, 3)
        p2.mirror('y')
        self.assertEqual(list(p2), [1, -2, 3])

        p3 = Point(1, 2, 3)
        p3.mirror('z')
        self.assertEqual(list(p3), [1, 2, -3])

        p4 = Point(1, 2, 3)
        p4.mirror('X')
        self.assertEqual(list(p4), [-1, 2, 3])

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
        self.assertEqual(list(s1.p0), [-1, 2, 3])
        self.assertEqual(list(s1.p1), [-4, 5, 6])

        s2 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s2.mirror('y')
        self.assertEqual(list(s2.p0), [1, -2, 3])
        self.assertEqual(list(s2.p1), [4, -5, 6])

        s3 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s3.mirror('z')
        self.assertEqual(list(s3.p0), [1, 2, -3])
        self.assertEqual(list(s3.p1), [4, 5, -6])

        s4 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s4.mirror('Y')
        self.assertEqual(list(s4.p0), [1, -2, 3])
        self.assertEqual(list(s4.p1), [4, -5, 6])

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
        self.assertEqual(list(b1[1]), [-1, 1, 0])
        self.assertEqual(list(b1[2]), [-3, 1, 0])
        self.assertEqual(list(b1[3]), [-4, 0, 0])

        b2 = Bezier(Point(0, 0, 0), Point(1, 1, 0),
                    Point(3, 1, 0), Point(4, 0, 0))
        b2.mirror('y')
        self.assertEqual(list(b2[0]), [0, 0, 0])
        self.assertEqual(list(b2[1]), [1, -1, 0])
        self.assertEqual(list(b2[2]), [3, -1, 0])
        self.assertEqual(list(b2[3]), [4, 0, 0])

        b3 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b3.mirror('z')
        self.assertEqual(list(b3[0]), [1, 2, -3])
        self.assertEqual(list(b3[1]), [4, 5, -6])
        self.assertEqual(list(b3[2]), [7, 8, -9])
        self.assertEqual(list(b3[3]), [10, 11, -12])

        b4 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b4.mirror('Z')
        self.assertEqual(list(b4[0]), [1, 2, -3])

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
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), 2.0)
        self.assert_allclose(pp.x_at(1), -3.0)
        self.assert_allclose(pp.y_at(1), 4.0)

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
        self.assert_allclose(pp.z_at(0), -3.0)
        self.assert_allclose(pp.z_at(1), -6.0)

        pp.mirror('X')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.x_at(1), -4.0)

        with self.assertRaisesRegex(
                ValueError, "PointPad::mirror: axis must be 'x', 'y', or 'z'"):
            pp.mirror('w')


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
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.x1_at(0), -3.0)
        self.assert_allclose(sp.x0_at(1), -5.0)
        self.assert_allclose(sp.x1_at(1), -7.0)

        sp.mirror('y')
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.y1_at(0), -4.0)
        self.assert_allclose(sp.y0_at(1), -6.0)
        self.assert_allclose(sp.y1_at(1), -8.0)

    def test_mirror_3d(self):
        SegmentPad = self.SegmentPad

        sp = SegmentPad(ndim=3)
        sp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        sp.append(7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        sp.mirror('z')
        self.assert_allclose(sp.z0_at(0), -3.0)
        self.assert_allclose(sp.z1_at(0), -6.0)
        self.assert_allclose(sp.z0_at(1), -9.0)
        self.assert_allclose(sp.z1_at(1), -12.0)

        sp.mirror('X')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.x1_at(0), -4.0)

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
        self.assert_allclose(list(cp.x0), [-1, 1])
        self.assert_allclose(list(cp.x1), [-4, 4])
        self.assert_allclose(list(cp.x2), [-7, 7])
        self.assert_allclose(list(cp.x3), [-10, 10])
        self.assert_allclose(list(cp.y0), [2, -2])
        self.assert_allclose(list(cp.z0), [3, -3])

        cp.mirror('y')
        self.assert_allclose(list(cp.y0), [-2, 2])
        self.assert_allclose(list(cp.y1), [-5, 5])
        self.assert_allclose(list(cp.y2), [-8, 8])
        self.assert_allclose(list(cp.y3), [-11, 11])

        cp.mirror('Z')
        self.assert_allclose(list(cp.z0), [-3, 3])
        self.assert_allclose(list(cp.z1), [-6, 6])
        self.assert_allclose(list(cp.z2), [-9, 9])
        self.assert_allclose(list(cp.z3), [-12, 12])

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
