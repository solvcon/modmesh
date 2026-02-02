# Copyright (c) 2026, An-Chi Liu <phy.tiger@gmail.com>
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
import modmesh as mm
from modmesh.testing import TestBase as ModMeshTB


class TrapezoidalDecomposerTB(ModMeshTB):

    def _make_2d_points(self, *coords):
        """Helper to create Point objects from 2D coordinates.

        Args:
            *coords: Variable number of (x, y) tuples

        Returns:
            List of Point objects with z=0.0
        """
        return [self.Point(x, y, 0.0) for x, y in coords]

    def test_simple_rectangle(self):
        """Test decomposition of a simple rectangle."""
        decomposer = self.TrapezoidalDecomposer(2)

        rectangle = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, rectangle)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 2.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.x2(0), 2.0)
        self.assert_allclose(trapezoids.y2(0), 1.0)
        self.assert_allclose(trapezoids.x3(0), 0.0)
        self.assert_allclose(trapezoids.y3(0), 1.0)

    def test_simple_triangle(self):
        """Test decomposition of a simple triangle."""
        decomposer = self.TrapezoidalDecomposer(2)

        triangle = self._make_2d_points((0.0, 0.0), (2.0, 0.0), (1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, triangle)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 2.0)

        self.assert_allclose(trapezoids.y2(0), 2.0)
        self.assert_allclose(trapezoids.y3(0), 2.0)
        self.assert_allclose(trapezoids.x2(0), 1.0)
        self.assert_allclose(trapezoids.x3(0), 1.0)

    def test_trapezoid_shape(self):
        """Test decomposition of a trapezoid shape."""
        decomposer = self.TrapezoidalDecomposer(2)

        trapezoid = self._make_2d_points(
            (0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, trapezoid)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.y2(0), 2.0)
        self.assert_allclose(trapezoids.y3(0), 2.0)

        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 4.0)
        self.assert_allclose(trapezoids.x2(0), 3.0)
        self.assert_allclose(trapezoids.x3(0), 1.0)

    def test_pentagon(self):
        """Test decomposition of a pentagon."""
        decomposer = self.TrapezoidalDecomposer(2)

        pentagon = self._make_2d_points(
            (0.0, 0.0), (3.0, 0.0), (4.0, 2.0), (1.5, 3.0), (-1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, pentagon)

        self.assertEqual(decomposer.num_trapezoids(0), 2)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_concave_polygon(self):
        """Test decomposition of a concave polygon."""
        decomposer = self.TrapezoidalDecomposer(2)

        concave = self._make_2d_points(
            (0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 1.5), (0.0, 3.0))

        begin_index, end_index = decomposer.decompose(0, concave)

        self.assertEqual(decomposer.num_trapezoids(0), 3)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 3)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 3)

    def test_multiple_polygons(self):
        """Test decomposition of multiple polygons."""
        decomposer = self.TrapezoidalDecomposer(2)

        square1 = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        square2 = self._make_2d_points(
            (2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))

        begin1, end1 = decomposer.decompose(0, square1)
        begin2, end2 = decomposer.decompose(1, square2)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(decomposer.num_trapezoids(1), 1)

        self.assertEqual(begin1, 0)
        self.assertEqual(end1, 1)
        self.assertEqual(begin2, 1)
        self.assertEqual(end2, 2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_cached_decomposition(self):
        """Test that decomposition results are cached."""
        decomposer = self.TrapezoidalDecomposer(2)

        square = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        begin1, end1 = decomposer.decompose(0, square)
        begin2, end2 = decomposer.decompose(0, square)

        self.assertEqual(begin1, begin2)
        self.assertEqual(end1, end2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

    def test_clear(self):
        """Test clearing decomposition results."""
        decomposer = self.TrapezoidalDecomposer(2)

        square = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        decomposer.decompose(0, square)
        self.assertEqual(decomposer.num_trapezoids(0), 1)

        decomposer.clear()

        self.assertEqual(decomposer.num_trapezoids(0), 0)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 0)

    def test_horizontal_edges_ignored(self):
        """Test that horizontal edges are ignored during decomposition."""
        decomposer = self.TrapezoidalDecomposer(2)

        shape_with_horizontal = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (2.0, 1.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, shape_with_horizontal)

        self.assertEqual(decomposer.num_trapezoids(0), 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

    def test_floating_point_precision(self):
        """Test decomposition with different floating point precisions."""
        decomposer_fp32 = mm.TrapezoidalDecomposerFp32(2)
        decomposer_fp64 = self.TrapezoidalDecomposer(2)

        triangle_fp32 = [
            mm.Point3dFp32(0.0, 0.0, 0.0),
            mm.Point3dFp32(1.0, 0.0, 0.0),
            mm.Point3dFp32(0.5, 1.0, 0.0)
        ]

        triangle_fp64 = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (0.5, 1.0))

        begin32, end32 = decomposer_fp32.decompose(0, triangle_fp32)
        begin64, end64 = decomposer_fp64.decompose(0, triangle_fp64)

        self.assertEqual(decomposer_fp32.num_trapezoids(0),
                         decomposer_fp64.num_trapezoids(0))

    def test_complex_polygon_with_many_vertices(self):
        """Test decomposition of a complex polygon with many vertices."""
        decomposer = self.TrapezoidalDecomposer(2)

        octagon = self._make_2d_points(
            (1.0, 0.0), (2.0, 0.0), (3.0, 1.0), (3.0, 2.0),
            (2.0, 3.0), (1.0, 3.0), (0.0, 2.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, octagon)

        self.assertEqual(decomposer.num_trapezoids(0), 3)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 3)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 3)

    def test_l_shaped_polygon(self):
        """Test decomposition of an L-shaped polygon."""
        decomposer = self.TrapezoidalDecomposer(2)

        l_shape = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
            (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))

        begin_index, end_index = decomposer.decompose(0, l_shape)

        self.assertEqual(decomposer.num_trapezoids(0), 2)
        self.assertEqual(begin_index, 0)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_narrow_spike(self):
        """Test decomposition of a narrow spike."""
        decomposer = self.TrapezoidalDecomposer(2)

        spike = self._make_2d_points((0.0, 0.0), (1.0, 0.0), (0.5, 10.0))

        begin_index, end_index = decomposer.decompose(0, spike)

        self.assertEqual(decomposer.num_trapezoids(0), 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)


class TrapezoidalDecomposerFp32TC(TrapezoidalDecomposerTB, unittest.TestCase):

    def setUp(self):
        self.Point = mm.Point3dFp32
        self.TrapezoidalDecomposer = mm.TrapezoidalDecomposerFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class TrapezoidalDecomposerFp64TC(TrapezoidalDecomposerTB, unittest.TestCase):

    def setUp(self):
        self.Point = mm.Point3dFp64
        self.TrapezoidalDecomposer = mm.TrapezoidalDecomposerFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
