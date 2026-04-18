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


class WorldTB(testing.TestBase):

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


class WorldShapeTC(unittest.TestCase):
    """Shape registry: add, translate, remove, clear."""

    def setUp(self):
        self.w = modmesh.WorldFp64()

    def test_add_triangle(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nsegment, 3)
        self.assertEqual(self.w.shape_type_of(sid), "triangle")

    def test_add_multiple(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        self.assertEqual(self.w.nshape, 2)
        self.assertEqual(self.w.nsegment, 6)

    def test_ids_are_unique(self):
        s0 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        s1 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertNotEqual(s0, s1)

    def test_translate(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.translate_shape(sid, 10, 20)
        seg = self.w.segment(0)
        self.assertAlmostEqual(seg.x0, 10.0)
        self.assertAlmostEqual(seg.y0, 20.0)
        self.assertAlmostEqual(seg.x1, 11.0)
        self.assertAlmostEqual(seg.y1, 20.0)

    def test_translate_isolates_shapes(self):
        s0 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(0, 0, 2, 0, 0, 2)
        self.w.translate_shape(s0, 10, 10)
        self.assertAlmostEqual(self.w.segment(0).x0, 10.0)
        self.assertAlmostEqual(self.w.segment(3).x0, 0.0)

    def test_remove(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        self.assertEqual(self.w.nshape, 0)

    def test_remove_nonexistent_raises(self):
        with self.assertRaises(IndexError):
            self.w.remove_shape(999)

    def test_remove_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.remove_shape(sid)

    def test_translate_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.translate_shape(sid, 1, 1)

    def test_shape_type_of_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.shape_type_of(sid)

    def test_clear(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        self.w.add_segment(
            s=modmesh.Segment3dFp64(
                modmesh.Point3dFp64(0, 0),
                modmesh.Point3dFp64(1, 1),
            )
        )
        self.w.clear()
        self.assertEqual(self.w.nshape, 0)
        self.assertEqual(self.w.nsegment, 0)


class WorldViewportTC(unittest.TestCase):
    """R-tree spatial index and viewport query."""

    def setUp(self):
        self.w = modmesh.WorldFp64()

    def test_all_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        ids = self.w.query_visible(-1, -1, 10, 10)
        self.assertEqual(len(ids), 2)

    def test_partial_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(100, 100, 101, 100, 100, 101)
        ids = self.w.query_visible(-1, -1, 2, 2)
        self.assertEqual(len(ids), 1)

    def test_none_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertEqual(len(self.w.query_visible(50, 50, 60, 60)), 0)

    def test_visible_after_translate(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.translate_shape(sid, 200, 200)
        self.assertEqual(len(self.w.query_visible(-1, -1, 2, 2)), 0)
        self.assertEqual(
            len(self.w.query_visible(199, 199, 202, 202)), 1)

    def test_visible_after_remove(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        self.assertEqual(len(self.w.query_visible(-1, -1, 10, 10)), 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
