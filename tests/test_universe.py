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


class ModMeshTB:

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-12
        return np.testing.assert_allclose(*args, **kw)


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


class Vector3dTB(ModMeshTB):

    def test_construct(self):
        Vector3d = self.kls

        # Construct using positional arguments
        vec = Vector3d(1, 2, 3)
        self.assertEqual(vec.x, 1.0)
        self.assertEqual(vec.y, 2.0)
        self.assertEqual(vec.z, 3.0)

        # Construct using keyword arguments
        vec = Vector3d(x=2.2, y=5.8, z=-9.22)
        self.assert_allclose(vec, [2.2, 5.8, -9.22])
        self.assert_allclose(vec[0], 2.2)
        self.assert_allclose(vec[1], 5.8)
        self.assert_allclose(vec[2], -9.22)
        self.assertEqual(len(vec), 3)

        # Range error in C++
        with self.assertRaisesRegex(IndexError, "Vector3d: i 3 >= size 3"):
            vec[3]

    def test_fill(self):
        Vector3d = self.kls

        vec = Vector3d(1, 2, 3)
        vec.fill(10.0)
        self.assertEqual(list(vec), [10, 10, 10])


class Vector3dFp32TC(Vector3dTB, unittest.TestCase):

    def setUp(self):
        self.kls = modmesh.Vector3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Vector3dFp32, self.kls)


class Vector3dFp64TC(Vector3dTB, unittest.TestCase):

    def setUp(self):
        self.kls = modmesh.Vector3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.Vector3dFp64, self.kls)


class Bezier3dTB(ModMeshTB):

    def test_control_points(self):
        Vector3d = self.vkls
        Bezier3d = self.bkls

        # Create a cubic Bezier curve
        bzr = Bezier3d(
            [Vector3d(0, 0, 0), Vector3d(1, 1, 0), Vector3d(3, 1, 0),
             Vector3d(4, 0, 0)])
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

        bzr.control_points = [Vector3d(4, 0, 0), Vector3d(3, 1, 0),
                              Vector3d(1, 1, 0), Vector3d(0, 0, 0)]
        self.assertEqual(list(bzr.control_points[0]), [4, 0, 0])
        self.assertEqual(list(bzr.control_points[1]), [3, 1, 0])
        self.assertEqual(list(bzr.control_points[2]), [1, 1, 0])
        self.assertEqual(list(bzr.control_points[3]), [0, 0, 0])

        with self.assertRaisesRegex(
                IndexError,
                "Bezier3d.control_points: len\\(points\\) 3 != ncontrol 4"):
            bzr.control_points = [Vector3d(3, 1, 0), Vector3d(1, 1, 0),
                                  Vector3d(0, 0, 0)]
        with self.assertRaisesRegex(
                IndexError,
                "Bezier3d.control_points: len\\(points\\) 5 != ncontrol 4"):
            bzr.control_points = [Vector3d(4, 0, 0), Vector3d(3, 1, 0),
                                  Vector3d(1, 1, 0), Vector3d(0, 0, 0),
                                  Vector3d(0, 0, 0)]

        # Locus point API
        self.assertEqual(len(bzr.locus_points), 0)

    def test_local_points(self):
        Vector3d = self.vkls
        Bezier3d = self.bkls

        b = Bezier3d(
            [Vector3d(0, 0, 0), Vector3d(1, 1, 0), Vector3d(3, 1, 0),
             Vector3d(4, 0, 0)])
        self.assertEqual(len(b.control_points), 4)
        self.assertEqual(len(b.locus_points), 0)

        b.sample(5)
        self.assertEqual(len(b.locus_points), 5)
        self.assert_allclose([list(p) for p in b.locus_points],
                             [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0],
                              [2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0],
                              [4.0, 0.0, 0.0]])

        b.sample(9)
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
        self.vkls = modmesh.Vector3dFp32
        self.bkls = modmesh.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class Bezier3dFp64TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp32
        self.bkls = modmesh.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
