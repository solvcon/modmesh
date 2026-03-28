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
from modmesh import testing


class Triangle3dTB(testing.TestBase):

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
        t = modmesh.Triangle3dFp32(modmesh.Point3dFp32(1.5, 2.5, 3.5),
                                   modmesh.Point3dFp32(4.5, 5.5, 6.5),
                                   modmesh.Point3dFp32(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp32(Point3dFp32(1.5, 2.5, 3.5), "
                  "Point3dFp32(4.5, 5.5, 6.5), "
                  "Point3dFp32(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden, vars(modmesh))
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
        t = modmesh.Triangle3dFp64(modmesh.Point3dFp64(1.5, 2.5, 3.5),
                                   modmesh.Point3dFp64(4.5, 5.5, 6.5),
                                   modmesh.Point3dFp64(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp64(Point3dFp64(1.5, 2.5, 3.5), "
                  "Point3dFp64(4.5, 5.5, 6.5), "
                  "Point3dFp64(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden, vars(modmesh))
        self.assertEqual(t, e)


class TrianglePadTB(testing.TestBase):

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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
