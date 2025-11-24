# Copyright (c) 2025, An-Chi Liu <phy.tiger@gmail.com>
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


class Polygon3dTB(ModMeshTB):

    def test_construct_from_segment_pad_2d(self):
        """Test constructing Polygon3d from a 2D SegmentPad."""
        segment_pad = self.SegmentPad(ndim=2)
        segment_pad.append(0.0, 0.0, 1.0, 0.0)
        segment_pad.append(1.0, 0.0, 1.0, 1.0)
        segment_pad.append(1.0, 1.0, 0.0, 1.0)
        segment_pad.append(0.0, 1.0, 0.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        self.assertEqual(polygon.ndim, 2)
        self.assertEqual(polygon.size, 4)

        segment0 = polygon.get(0)
        self.assert_allclose([segment0.x0, segment0.y0], [0.0, 0.0])
        self.assert_allclose([segment0.x1, segment0.y1], [1.0, 0.0])

        segment1 = polygon.get(1)
        self.assert_allclose([segment1.x0, segment1.y0], [1.0, 0.0])
        self.assert_allclose([segment1.x1, segment1.y1], [1.0, 1.0])

    def test_construct_from_segment_pad_3d(self):
        """Test constructing Polygon3d from a 3D SegmentPad."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad.append(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        segment_pad.append(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        segment_pad.append(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        self.assertEqual(polygon.ndim, 3)
        self.assertEqual(polygon.size, 4)

        segment0 = polygon.get(0)
        self.assert_allclose([segment0.x0, segment0.y0, segment0.z0],
                             [0.0, 0.0, 0.0])
        self.assert_allclose([segment0.x1, segment0.y1, segment0.z1],
                             [1.0, 0.0, 0.0])

    def test_construct_from_curve_pad(self):
        """Test constructing Polygon3d from a CurvePad by sampling."""
        curve_pad = self.CurvePad(ndim=3)

        p0 = self.Point(0.0, 0.0, 0.0)
        p1 = self.Point(0.5, 0.5, 0.0)
        p2 = self.Point(1.0, 0.5, 0.0)
        p3 = self.Point(1.0, 0.0, 0.0)
        curve_pad.append(p0, p1, p2, p3)

        sample_length = 0.2
        polygon = self.Polygon3d(curve_pad, sample_length)

        self.assertEqual(polygon.ndim, 3)
        self.assertGreater(polygon.size, 0)

        segments = polygon.segments
        self.assertIsNotNone(segments)
        self.assertEqual(segments.ndim, 3)
        self.assertEqual(len(segments), polygon.size)

    def test_construct_from_both_segment_and_curve(self):
        """Test constructing Polygon3d from both SegmentPad and CurvePad."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad.append(1.0, 0.0, 0.0, 2.0, 0.0, 0.0)

        curve_pad = self.CurvePad(ndim=3)
        p0 = self.Point(2.0, 0.0, 0.0)
        p1 = self.Point(2.5, 0.5, 0.0)
        p2 = self.Point(3.0, 0.5, 0.0)
        p3 = self.Point(3.0, 0.0, 0.0)
        curve_pad.append(p0, p1, p2, p3)

        sample_length = 0.2
        polygon = self.Polygon3d(segment_pad, curve_pad, sample_length)

        self.assertEqual(polygon.ndim, 3)
        self.assertGreater(polygon.size, 2)

        segments = polygon.segments
        first_segment = segments[0]
        self.assert_allclose(
            [first_segment.x0, first_segment.y0, first_segment.z0],
            [0.0, 0.0, 0.0])
        self.assert_allclose(
            [first_segment.x1, first_segment.y1, first_segment.z1],
            [1.0, 0.0, 0.0])

        second_segment = segments[1]
        self.assert_allclose(
            [second_segment.x0, second_segment.y0, second_segment.z0],
            [1.0, 0.0, 0.0])
        self.assert_allclose(
            [second_segment.x1, second_segment.y1, second_segment.z1],
            [2.0, 0.0, 0.0])

    def test_bound_box(self):
        """Test bounding box calculation."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(-1.0, -2.0, -3.0, 4.0, 5.0, 6.0)
        segment_pad.append(0.0, 0.0, 0.0, 2.0, 3.0, 1.0)

        polygon = self.Polygon3d(segment_pad)
        bbox = polygon.calc_bound_box()

        self.assert_allclose(bbox.min_x, -1.0)
        self.assert_allclose(bbox.min_y, -2.0)
        self.assert_allclose(bbox.min_z, -3.0)
        self.assert_allclose(bbox.max_x, 4.0)
        self.assert_allclose(bbox.max_y, 5.0)
        self.assert_allclose(bbox.max_z, 6.0)

    def test_empty_polygon_bound_box(self):
        """Test bounding box of empty polygon."""
        segment_pad = self.SegmentPad(ndim=3)
        polygon = self.Polygon3d(segment_pad)

        bbox = polygon.calc_bound_box()
        self.assert_allclose(bbox.min_x, 0.0)
        self.assert_allclose(bbox.min_y, 0.0)
        self.assert_allclose(bbox.min_z, 0.0)
        self.assert_allclose(bbox.max_x, 0.0)
        self.assert_allclose(bbox.max_y, 0.0)
        self.assert_allclose(bbox.max_z, 0.0)

    def test_segment_access(self):
        """Test accessing individual segments."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad.append(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        segment_pad.append(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        self.assertEqual(polygon.size, 3)

        seg0 = polygon.get(0)
        self.assert_allclose([seg0.x0, seg0.y0, seg0.z0], [0.0, 0.0, 0.0])
        self.assert_allclose([seg0.x1, seg0.y1, seg0.z1], [1.0, 0.0, 0.0])

        seg1 = polygon.get_at(1)
        self.assert_allclose([seg1.x0, seg1.y0, seg1.z0], [1.0, 0.0, 0.0])
        self.assert_allclose([seg1.x1, seg1.y1, seg1.z1], [1.0, 1.0, 0.0])

        seg2 = polygon.get(2)
        self.assert_allclose([seg2.x0, seg2.y0, seg2.z0], [1.0, 1.0, 0.0])
        self.assert_allclose([seg2.x1, seg2.y1, seg2.z1], [0.0, 1.0, 0.0])

    def test_equality_operators(self):
        """Test equality and inequality operators."""
        segment_pad1 = self.SegmentPad(ndim=3)
        segment_pad1.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad1.append(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)

        segment_pad2 = self.SegmentPad(ndim=3)
        segment_pad2.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad2.append(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)

        segment_pad3 = self.SegmentPad(ndim=3)
        segment_pad3.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad3.append(2.0, 0.0, 0.0, 2.0, 1.0, 0.0)

        polygon1 = self.Polygon3d(segment_pad1)
        polygon2 = self.Polygon3d(segment_pad2)
        polygon3 = self.Polygon3d(segment_pad3)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1 != polygon2)
        self.assertTrue(polygon1 != polygon3)
        self.assertFalse(polygon1 == polygon3)

    def test_search_segments(self):
        """Test RTree-based segment searching."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        segment_pad.append(2.0, 0.0, 0.0, 3.0, 0.0, 0.0)
        segment_pad.append(5.0, 5.0, 0.0, 6.0, 6.0, 0.0)
        segment_pad.append(10.0, 10.0, 0.0, 11.0, 11.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        BoundBox = (mm.BoundBox3dFp32 if self.dtype == 'float32'
                    else mm.BoundBox3dFp64)
        search_box = BoundBox(-0.5, -0.5, -0.5, 3.5, 0.5, 0.5)
        results = polygon.search_segments(search_box)

        self.assertEqual(len(results), 2)

        search_box2 = BoundBox(4.5, 4.5, -0.5, 6.5, 6.5, 0.5)
        results2 = polygon.search_segments(search_box2)
        self.assertEqual(len(results2), 1)

        search_box3 = BoundBox(100.0, 100.0, 0.0, 200.0, 200.0, 0.0)
        results3 = polygon.search_segments(search_box3)
        self.assertEqual(len(results3), 0)

    def test_rebuild_rtree(self):
        """Test rebuilding the RTree."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        self.assertEqual(polygon.size, 1)

        polygon.rebuild_rtree()

        BoundBox = (mm.BoundBox3dFp32 if self.dtype == 'float32'
                    else mm.BoundBox3dFp64)
        search_box = BoundBox(-0.5, -0.5, -0.5, 1.5, 0.5, 0.5)
        results = polygon.search_segments(search_box)
        self.assertEqual(len(results), 1)

    def test_square_polygon_2d(self):
        """Test creating a complete square polygon in 2D."""
        segment_pad = self.SegmentPad(ndim=2)
        segment_pad.append(0.0, 0.0, 1.0, 0.0)
        segment_pad.append(1.0, 0.0, 1.0, 1.0)
        segment_pad.append(1.0, 1.0, 0.0, 1.0)
        segment_pad.append(0.0, 1.0, 0.0, 0.0)

        polygon = self.Polygon3d(segment_pad)

        self.assertEqual(polygon.size, 4)
        self.assertEqual(polygon.ndim, 2)

        bbox = polygon.calc_bound_box()
        self.assert_allclose(bbox.min_x, 0.0)
        self.assert_allclose(bbox.min_y, 0.0)
        self.assert_allclose(bbox.max_x, 1.0)
        self.assert_allclose(bbox.max_y, 1.0)

    def test_curved_polygon(self):
        """Test creating a polygon with curved edges."""
        curve_pad = self.CurvePad(ndim=3)

        p0 = self.Point(0.0, 0.0, 0.0)
        p1 = self.Point(0.0, 1.0, 0.0)
        p2 = self.Point(1.0, 1.0, 0.0)
        p3 = self.Point(1.0, 0.0, 0.0)
        curve_pad.append(p0, p1, p2, p3)

        p0 = self.Point(1.0, 0.0, 0.0)
        p1 = self.Point(2.0, 0.0, 0.0)
        p2 = self.Point(2.0, 1.0, 0.0)
        p3 = self.Point(2.0, 2.0, 0.0)
        curve_pad.append(p0, p1, p2, p3)

        sample_length = 0.3
        polygon = self.Polygon3d(curve_pad, sample_length)

        self.assertEqual(polygon.ndim, 3)
        self.assertGreater(polygon.size, 2)

        bbox = polygon.calc_bound_box()
        self.assertLessEqual(bbox.min_x, 0.0)
        self.assertGreaterEqual(bbox.max_x, 1.0)

    def test_mixed_straight_and_curved_edges(self):
        """Test polygon with both straight segments and curved edges."""
        segment_pad = self.SegmentPad(ndim=3)
        segment_pad.append(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        curve_pad = self.CurvePad(ndim=3)
        p0 = self.Point(1.0, 0.0, 0.0)
        p1 = self.Point(1.5, 0.5, 0.0)
        p2 = self.Point(1.5, 1.0, 0.0)
        p3 = self.Point(1.0, 1.0, 0.0)
        curve_pad.append(p0, p1, p2, p3)

        sample_length = 0.2
        polygon = self.Polygon3d(segment_pad, curve_pad, sample_length)

        self.assertEqual(polygon.ndim, 3)
        self.assertGreater(polygon.size, 1)

        first_segment = polygon.segments[0]
        self.assert_allclose(
            [first_segment.x0, first_segment.y0, first_segment.z0],
            [0.0, 0.0, 0.0])


class Polygon3dFp32TC(Polygon3dTB, unittest.TestCase):

    dtype = 'float32'
    Point = mm.Point3dFp32
    SegmentPad = mm.SegmentPadFp32
    CurvePad = mm.CurvePadFp32
    Polygon3d = mm.Polygon3dFp32
    SimpleArray = mm.SimpleArrayFloat32


class Polygon3dFp64TC(Polygon3dTB, unittest.TestCase):

    dtype = 'float64'
    Point = mm.Point3dFp64
    SegmentPad = mm.SegmentPadFp64
    CurvePad = mm.CurvePadFp64
    Polygon3d = mm.Polygon3dFp64
    SimpleArray = mm.SimpleArrayFloat64
