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


class Vector3dTB(ModMeshTB):

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
        with self.assertRaisesRegex(IndexError, "Vector3d: i 3 >= size 3"):
            vec[3]

    def test_fill(self):
        Vector = self.kls

        vec = Vector(1, 2, 3)
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


class Edge3dTB(ModMeshTB):

    def test_construct(self):
        Vector = self.vkls
        Edge = self.ekls

        e = Edge(x0=0, y0=0, z0=0, x1=1, y1=1, z1=1)
        self.assertEqual(len(e), 2)
        self.assertEqual(tuple(e.v0), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(e.v1), (1.0, 1.0, 1.0))

        e.v0 = Vector(x=3, y=7, z=0)
        e.v1 = Vector(x=-1, y=-4, z=9)
        self.assertEqual(e.x0, 3)
        self.assertEqual(e.y0, 7)
        self.assertEqual(e.z0, 0)
        self.assertEqual(e.x1, -1)
        self.assertEqual(e.y1, -4)
        self.assertEqual(e.z1, 9)

        e = Edge(Vector(x=3.1, y=7.4, z=0.6), Vector(x=-1.2, y=-4.1, z=9.2))
        self.assert_allclose(tuple(e.v0), (3.1, 7.4, 0.6))
        self.assert_allclose(tuple(e.v1), (-1.2, -4.1, 9.2))


class Edge3dFp32TC(Edge3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp32
        self.ekls = modmesh.Edge3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class Edge3dFp64TC(Edge3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp64
        self.ekls = modmesh.Edge3dFp64

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
        self.vkls = modmesh.Vector3dFp32
        self.bkls = modmesh.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class Bezier3dFp64TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp64
        self.bkls = modmesh.Bezier3dFp64

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
        self.assertEqual(v, w.vertex(0))
        self.assertEqual(w.nvertex, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(vertex\\) i 1 >= size 1"):
            w.vertex(1)

        # Add a vertex by coordinate
        v = w.add_vertex(3.1415, 3.1416, 3.1417)
        self.assert_allclose(list(v), [3.1415, 3.1416, 3.1417])
        self.assert_allclose(list(w.vertex(1)), [3.1415, 3.1416, 3.1417])
        self.assertEqual(v, w.vertex(1))
        self.assertEqual(w.nvertex, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(vertex\\) i 2 >= size 2"):
            w.vertex(2)


class WorldFp32TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp32
        self.wkls = modmesh.WorldFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp32, self.wkls)


class WorldFp64TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.vkls = modmesh.Vector3dFp64
        self.wkls = modmesh.WorldFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(modmesh.WorldFp64, self.wkls)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
