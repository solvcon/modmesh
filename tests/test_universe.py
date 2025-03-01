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
        Vector = self.kls

        # Construct using positional arguments
        vec = Vector(1, 2, 3)
        self.assertEqual(vec.x, 1.0)
        self.assertEqual(vec.y, 2.0)
        self.assertEqual(vec.z, 3.0)

        # Construct using keyword arguments
        vec = Vector(x=2.2, y=5.8, z=-9.22)
        self.assert_allclose(vec, [2.2, 5.8, -9.22])
        self.assert_allclose(vec[0], 2.2)
        self.assert_allclose(vec[1], 5.8)
        self.assert_allclose(vec[2], -9.22)
        self.assertEqual(len(vec), 3)

        # Range error in C++
        with self.assertRaisesRegex(IndexError, "Point3d: i 3 >= size 3"):
            vec[3]

    def test_fill(self):
        Vector = self.kls

        vec = Vector(1, 2, 3)
        vec.fill(10.0)
        self.assertEqual(list(vec), [10, 10, 10])

    def test_arithmetic(self):
        Point = self.kls
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


class Point3dFp32TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.kls = modmesh.Point3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Point3dFp32, self.kls)


class Point3dFp64TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.kls = modmesh.Point3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Point3dFp64, self.kls)


class Segment3dTB(ModMeshTB):

    def test_construct(self):
        Point = self.vkls
        Segment = self.gkls

        s = Segment(x0=0, y0=0, z0=0, x1=1, y1=1, z1=1)
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


class Segment3dFp32TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp32
        self.gkls = modmesh.Segment3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class Segment3dFp64TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp64
        self.gkls = modmesh.Segment3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class Bezier3dTB(ModMeshTB):

    def test_control_points(self):
        Vector = self.vkls
        Bezier = self.bkls

        # Create a cubic Bezier curve
        bzr = Bezier(
            [Vector(0, 0, 0), Vector(1, 1, 0), Vector(3, 1, 0),
             Vector(4, 0, 0)])
        self.assertEqual(len(bzr), 4)
        self.assertEqual(list(bzr[0]), [0, 0, 0])
        self.assertEqual(list(bzr[1]), [1, 1, 0])
        self.assertEqual(list(bzr[2]), [3, 1, 0])
        self.assertEqual(list(bzr[3]), [4, 0, 0])

        # Range error in C++
        with self.assertRaisesRegex(IndexError,
                                    "Bezier3d: \\(control\\) i 4 >= size 4"):
            bzr[4]

        # Control point API
        self.assertEqual(len(bzr.control_points), 4)
        self.assertEqual(list(bzr.control_points[0]), [0, 0, 0])
        self.assertEqual(list(bzr.control_points[1]), [1, 1, 0])
        self.assertEqual(list(bzr.control_points[2]), [3, 1, 0])
        self.assertEqual(list(bzr.control_points[3]), [4, 0, 0])

        bzr.control_points = [Vector(4, 0, 0), Vector(3, 1, 0),
                              Vector(1, 1, 0), Vector(0, 0, 0)]
        self.assertEqual(list(bzr.control_points[0]), [4, 0, 0])
        self.assertEqual(list(bzr.control_points[1]), [3, 1, 0])
        self.assertEqual(list(bzr.control_points[2]), [1, 1, 0])
        self.assertEqual(list(bzr.control_points[3]), [0, 0, 0])

        with self.assertRaisesRegex(
                IndexError,
                "Bezier3d.control_points: len\\(points\\) 3 != ncontrol 4"):
            bzr.control_points = [Vector(3, 1, 0), Vector(1, 1, 0),
                                  Vector(0, 0, 0)]
        with self.assertRaisesRegex(
                IndexError,
                "Bezier3d.control_points: len\\(points\\) 5 != ncontrol 4"):
            bzr.control_points = [Vector(4, 0, 0), Vector(3, 1, 0),
                                  Vector(1, 1, 0), Vector(0, 0, 0),
                                  Vector(0, 0, 0)]

        # Locus point API
        self.assertEqual(len(bzr.locus_points), 0)

    def test_local_points(self):
        Vector = self.vkls
        Bezier = self.bkls

        b = Bezier(
            [Vector(0, 0, 0), Vector(1, 1, 0), Vector(3, 1, 0),
             Vector(4, 0, 0)])
        self.assertEqual(len(b.control_points), 4)
        self.assertEqual(b.nlocus, 0)
        self.assertEqual(len(b.locus_points), 0)

        b.sample(5)
        self.assertEqual(b.nlocus, 5)
        self.assertEqual(len(b.locus_points), 5)
        self.assert_allclose([list(p) for p in b.locus_points],
                             [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0],
                              [2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0],
                              [4.0, 0.0, 0.0]])

        b.sample(9)
        self.assertEqual(b.nlocus, 9)
        self.assertEqual(len(b.locus_points), 9)
        self.assert_allclose([list(p) for p in b.locus_points],
                             [[0.0, 0.0, 0.0], [0.41796875, 0.328125, 0.0],
                              [0.90625, 0.5625, 0.0],
                              [1.44140625, 0.703125, 0.0], [2.0, 0.75, 0.0],
                              [2.55859375, 0.703125, 0.0],
                              [3.09375, 0.5625, 0.0],
                              [3.58203125, 0.328125, 0.0], [4.0, 0.0, 0.0]])


class Bezier3dFp32TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp32
        self.bkls = modmesh.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class Bezier3dFp64TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp64
        self.bkls = modmesh.Bezier3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class PointPadTB(ModMeshTB):

    def test_ndim(self):
        pp2d = self.pkls(ndim=2)
        self.assertEqual(pp2d.ndim, 2)
        pp3d = self.pkls(ndim=3)
        self.assertEqual(pp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.pkls(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.pkls(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.pkls(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.pkls(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.pkls(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.pkls(ndim=4, nelem=5)

    def test_construct_2d(self):
        xarr = self.akls(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.akls(array=np.array([4, 5, 6], dtype=self.dtype))
        pp = self.pkls(x=xarr, y=yarr, clone=False)
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

        pp2 = self.pkls(ndim=2, nelem=3)
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
        xarr = self.akls(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.akls(array=np.array([4, 5, 6], dtype=self.dtype))
        zarr = self.akls(array=np.array([7, 8, 9], dtype=self.dtype))
        pp = self.pkls(x=xarr, y=yarr, z=zarr, clone=False)
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
            pp.set_at(3, self.vkls(0, 0, 0))

        # Test zero-copy writing
        pp.x[1] = 200.2
        pp.y[0] = -700.3
        pp.z[2] = 213.9
        self.assert_allclose(list(pp[0]), (1, -700.3, 7))
        self.assert_allclose(list(pp[1]), (200.2, 5, 8))
        self.assert_allclose(list(pp[2]), (3, 6, 213.9))

        pp2 = self.pkls(ndim=3, nelem=3)
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
        pp = self.pkls(ndim=2)
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
        pp = self.pkls(ndim=3)
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


class PointPadFp32TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.akls = modmesh.SimpleArrayFloat32
        self.vkls = modmesh.Point3dFp32
        self.pkls = modmesh.PointPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class PointPadFp64TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.akls = modmesh.SimpleArrayFloat64
        self.vkls = modmesh.Point3dFp64
        self.pkls = modmesh.PointPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class SegmentPadTB(ModMeshTB):

    def test_ndim(self):
        sp2d = self.skls(ndim=2)
        self.assertEqual(sp2d.ndim, 2)
        sp3d = self.skls(ndim=3)
        self.assertEqual(sp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.skls(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.skls(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.skls(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.skls(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.skls(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.skls(ndim=4, nelem=5)

    def test_construct_2d(self):
        x0arr = self.akls(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.akls(array=np.array([4, 5, 6], dtype=self.dtype))
        x1arr = self.akls(array=np.array([-1, -2, -3], dtype=self.dtype))
        y1arr = self.akls(array=np.array([-4, -5, -6], dtype=self.dtype))
        sp = self.skls(x0=x0arr, y0=y0arr, x1=x1arr, y1=y1arr, clone=False)
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

        sp2 = self.skls(ndim=2, nelem=3)
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
        x0arr = self.akls(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.akls(array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.akls(array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.akls(array=np.array([-1, -2, -3], dtype=self.dtype))
        y1arr = self.akls(array=np.array([-4, -5, -6], dtype=self.dtype))
        z1arr = self.akls(array=np.array([-7, -8, -9], dtype=self.dtype))
        sp = self.skls(x0=x0arr, y0=y0arr, z0=z0arr,
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
            sp.set_at(3, self.gkls(0, 0, 0, 0, 0, 0))

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

        sp2 = self.skls(ndim=3, nelem=3)
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
        sp = self.skls(ndim=2)
        self.assertEqual(sp.ndim, 2)
        self.assertEqual(len(sp), 0)
        sp.append(1.1, 2.2, 7.1, 8.2)
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        sp.append(1.1 * 3, 2.2 * 3, 7.1 * 3, 8.2 * 3)
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

    def test_append_3d(self):
        sp = self.skls(ndim=3)
        self.assertEqual(sp.ndim, 3)
        self.assertEqual(len(sp), 0)
        sp.append(1.1, 2.2, 3.3, 7.1, 8.2, 9.3)
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.z0_at(0), 3.3)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.z1_at(0), 9.3)
        sp.append(1.1 * 5, 2.2 * 5, 3.3 * 5, 7.1 * 5, 8.2 * 5, 9.3 * 5)
        self.assertEqual(len(sp), 2)
        self.assert_allclose(sp.x0_at(1), 1.1 * 5)
        self.assert_allclose(sp.y0_at(1), 2.2 * 5)
        self.assert_allclose(sp.z0_at(1), 3.3 * 5)
        self.assert_allclose(sp.x1_at(1), 7.1 * 5)
        self.assert_allclose(sp.y1_at(1), 8.2 * 5)
        self.assert_allclose(sp.z1_at(1), 9.3 * 5)
        sp.append(1.1 * 5.1, 2.2 * 5.1, 3.3 * 5.1, 7.1 * 5.1, 8.2 * 5.1,
                  9.3 * 5.1)
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


class SegmentPadFp32TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.akls = modmesh.SimpleArrayFloat32
        self.vkls = modmesh.Point3dFp32
        self.pkls = modmesh.PointPadFp32
        self.gkls = modmesh.Segment3dFp32
        self.skls = modmesh.SegmentPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class SegmentPadFp64TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.akls = modmesh.SimpleArrayFloat64
        self.vkls = modmesh.Point3dFp64
        self.pkls = modmesh.PointPadFp64
        self.gkls = modmesh.Segment3dFp64
        self.skls = modmesh.SegmentPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class WorldTB(ModMeshTB):

    def test_bezier(self):
        Vector = self.vkls
        World = self.wkls

        w = World()

        # Empty
        self.assertEqual(w.nbezier, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 0 >= size 0"):
            w.bezier(0)

        # Add Bezier curve
        b = w.add_bezier(
            [Vector(0, 0, 0), Vector(1, 1, 0), Vector(3, 1, 0),
             Vector(4, 0, 0)])
        self.assertEqual(w.nbezier, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 1 >= size 1"):
            w.bezier(1)

        # Check control points
        self.assertEqual(len(b), 4)
        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])

        # Check locus points
        self.assertEqual(b.nlocus, 0)
        self.assertEqual(len(b.locus_points), 0)
        b.sample(5)
        self.assertEqual(b.nlocus, 5)
        self.assertEqual(len(b.locus_points), 5)
        self.assert_allclose([list(p) for p in b.locus_points],
                             [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0],
                              [2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0],
                              [4.0, 0.0, 0.0]])

        # Confirm we worked on the internal instead of copy
        self.assertEqual(w.bezier(0).nlocus, 5)

    def test_vertex(self):
        Vector = self.vkls
        World = self.wkls

        w = World()

        # Empty
        self.assertEqual(w.nvertex, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(vertex\\) i 0 >= size 0"):
            w.vertex(0)

        # Add a vertex by object
        v = w.add_vertex(Vector(0, 1, 2))
        self.assertEqual(list(v), [0, 1, 2])
        self.assertEqual(list(w.vertex(0)), [0, 1, 2])
        self.assertIsNot(v, w.vertex(0))
        self.assertEqual(w.nvertex, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(vertex\\) i 1 >= size 1"):
            w.vertex(1)

        # Add a vertex by coordinate
        v = w.add_vertex(3.1415, 3.1416, 3.1417)
        self.assert_allclose(list(v), [3.1415, 3.1416, 3.1417])
        self.assert_allclose(list(w.vertex(1)), [3.1415, 3.1416, 3.1417])
        self.assertIsNot(v, w.vertex(1))
        self.assertEqual(w.nvertex, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(vertex\\) i 2 >= size 2"):
            w.vertex(2)

        # Add many vertices
        for it in range(10):
            w.add_vertex(3.1415 + it, 3.1416 + it, 3.1417 + it)
            self.assertEqual(w.nvertex, 2 + it + 1)


class WorldFp32TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp32
        self.wkls = modmesh.WorldFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp32, self.wkls)


class WorldFp64TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Point3dFp64
        self.wkls = modmesh.WorldFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp64, self.wkls)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
