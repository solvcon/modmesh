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

    def test_polygon_pad_basic(self):
        """Test PolygonPad with basic operations."""
        pad = self.PolygonPad(ndim=2)
        self.assertEqual(pad.ndim, 2)
        self.assertEqual(pad.num_polygons, 0)
        self.assertEqual(pad.num_points, 0)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        self.assertEqual(pad.num_polygons, 1)
        self.assertEqual(pad.num_points, 4)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

        node1 = polygon.get_node(1)
        self.assert_allclose([node1.x, node1.y], [1.0, 0.0])

    def test_polygon_handle_operations(self):
        """Test Polygon3d handle operations."""
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        self.assertEqual(polygon.ndim, 2)
        self.assertEqual(polygon.polygon_id, 0)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

        edge0 = polygon.get_edge(0)
        self.assert_allclose([edge0.x0, edge0.y0], [0.0, 0.0])
        self.assert_allclose([edge0.x1, edge0.y1], [1.0, 0.0])

        edge3 = polygon.get_edge(3)
        self.assert_allclose([edge3.x0, edge3.y0], [0.0, 1.0])
        self.assert_allclose([edge3.x1, edge3.y1], [0.0, 0.0])

    def test_right_hand_rule_counter_clockwise(self):
        """
        Test right-hand rule validation for counter-clockwise
        (positive area) polygon.
        """
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        area = polygon.compute_signed_area()
        self.assertGreater(area, 0.0)
        self.assertTrue(polygon.is_counter_clockwise())

    def test_right_hand_rule_clockwise(self):
        """Test right-hand rule: clockwise nodes should have negative area."""
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(0.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(1.0, 0.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        area = polygon.compute_signed_area()
        self.assertLess(area, 0.0)
        self.assertFalse(polygon.is_counter_clockwise())

    def test_multiple_polygons_in_pad(self):
        """Test storing multiple polygons in one PolygonPad."""
        pad = self.PolygonPad(ndim=2)

        square = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        triangle = [
            self.Point(2.0, 0.0, 0.0),
            self.Point(3.0, 0.0, 0.0),
            self.Point(2.5, 1.0, 0.0)
        ]

        poly1 = pad.add_polygon(square)
        poly2 = pad.add_polygon(triangle)

        self.assertEqual(pad.num_polygons, 2)
        self.assertEqual(pad.num_points, 7)

        self.assertEqual(poly1.nnode, 4)
        self.assertEqual(poly2.nnode, 3)

        self.assertGreater(poly1.compute_signed_area(), 0.0)
        self.assertGreater(poly2.compute_signed_area(), 0.0)

        retrieved_poly1 = pad.get_polygon(0)
        retrieved_poly2 = pad.get_polygon(1)

        self.assertEqual(retrieved_poly1.polygon_id, 0)
        self.assertEqual(retrieved_poly2.polygon_id, 1)

    def test_polygon_pad_from_segments(self):
        """Test adding polygon from SegmentPad."""
        segment_pad = self.SegmentPad(ndim=2)
        segment_pad.append(0.0, 0.0, 1.0, 0.0)
        segment_pad.append(1.0, 0.0, 1.0, 1.0)
        segment_pad.append(1.0, 1.0, 0.0, 1.0)
        segment_pad.append(0.0, 1.0, 0.0, 0.0)

        pad = self.PolygonPad(ndim=2)
        polygon = pad.add_polygon_from_segments(segment_pad)

        self.assertEqual(pad.num_polygons, 1)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

    def test_polygon_pad_rtree_search(self):
        """Test RTree search across multiple polygons in PolygonPad."""
        pad = self.PolygonPad(ndim=2)

        square1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        square2 = [
            self.Point(5.0, 5.0, 0.0),
            self.Point(6.0, 5.0, 0.0),
            self.Point(6.0, 6.0, 0.0),
            self.Point(5.0, 6.0, 0.0)
        ]

        pad.add_polygon(square1)
        pad.add_polygon(square2)

        BoundBox = (mm.BoundBox3dFp32 if self.dtype == 'float32'
                    else mm.BoundBox3dFp64)

        search_box1 = BoundBox(-0.5, -0.5, -0.5, 1.5, 1.5, 0.5)
        results1 = pad.search_segments(search_box1)
        self.assertEqual(len(results1), 4)

        search_box2 = BoundBox(4.5, 4.5, -0.5, 6.5, 6.5, 0.5)
        results2 = pad.search_segments(search_box2)
        self.assertEqual(len(results2), 4)

        search_box3 = BoundBox(-1.0, -1.0, -0.5, 7.0, 7.0, 0.5)
        results3 = pad.search_segments(search_box3)
        self.assertEqual(len(results3), 8)

    def test_polygon_equality_same_coordinates(self):
        """Test that polygons with same coordinates are equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1 != polygon2)

    def test_polygon_equality_different_coordinates(self):
        """Test that polygons with different coordinates are not equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertFalse(polygon1 == polygon2)
        self.assertTrue(polygon1 != polygon2)

    def test_polygon_equality_different_node_count(self):
        """Test that polygons with different node counts are not equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(0.5, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertFalse(polygon1 == polygon2)
        self.assertTrue(polygon1 != polygon2)

    def test_polygon_identity_same_pad_same_id(self):
        """Test identity: same pad and same polygon_id."""
        pad = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad.add_polygon(nodes)
        polygon2 = pad.get_polygon(0)

        self.assertTrue(polygon1.is_same(polygon2))
        self.assertFalse(polygon1.is_not_same(polygon2))

    def test_polygon_identity_different_pads(self):
        """Test identity: different pads with same coordinates."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes)
        polygon2 = pad2.add_polygon(nodes)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1.is_same(polygon2))
        self.assertTrue(polygon1.is_not_same(polygon2))

    def test_polygon_identity_same_pad_different_id(self):
        """Test identity: same pad but different polygon_id."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad.add_polygon(nodes1)
        polygon2 = pad.add_polygon(nodes2)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1.is_same(polygon2))
        self.assertTrue(polygon1.is_not_same(polygon2))

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_union_simple(self):
        """Test polygon boolean union with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3) - overlaps with first
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        # Call boolean_union - should not crash even if implementation is TODO
        try:
            result = pad.boolean_union(polygon1, polygon2)
            # Implementation is not complete yet, but should return a list
            self.assertIsInstance(result, self.PolygonPad)
        except NotImplementedError:
            # Expected if method raises NotImplementedError
            pass

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_intersection_simple(self):
        """Test polygon boolean intersection with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3) - overlaps with first
        # Intersection should be (1,1) to (2,2)
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        # Call boolean_intersection - should not crash even
        # if implementation is TODO
        try:
            result = pad.boolean_intersection(polygon1, polygon2)
            # Implementation is not complete yet, but should return a list
            self.assertIsInstance(result, self.PolygonPad)
        except NotImplementedError:
            # Expected if method raises NotImplementedError
            pass

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_difference_simple(self):
        """Test polygon boolean difference with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3) - overlaps with first
        # Difference (polygon1 - polygon2) should be L-shaped region
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        # Call boolean_difference - should not crash even
        # if implementation is TODO
        try:
            result = pad.boolean_difference(polygon1, polygon2)
            # Implementation is not complete yet, but should return a list
            self.assertIsInstance(result, self.PolygonPad)
        except NotImplementedError:
            # Expected if method raises NotImplementedError
            pass

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_union_non_overlapping(self):
        """Test polygon boolean union with two non-overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (1,1)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (2,2) to (3,3) - no overlap
        square2_nodes = [
            self.Point(2.0, 2.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(2.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        # Call boolean_union - should not crash
        try:
            result = pad.boolean_union(polygon1, polygon2)
            self.assertIsInstance(result, self.PolygonPad)
        except NotImplementedError:
            pass

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_intersection_non_overlapping(self):
        """Test polygon boolean intersection with two non-overlapping squares.
        """
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (1,1)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (2,2) to (3,3) - no overlap
        # Intersection should be empty
        square2_nodes = [
            self.Point(2.0, 2.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(2.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        # Call boolean_intersection - should not crash
        try:
            result = pad.boolean_intersection(polygon1, polygon2)
            self.assertIsInstance(result, self.PolygonPad)
        except NotImplementedError:
            pass

    # TODO: Verify and implement boolean operations
    # once the C++ side is complete.
    def test_boolean_operations_triangle(self):
        """Test polygon boolean operations with triangular polygons."""
        pad = self.PolygonPad(ndim=2)

        # Triangle 1: Right triangle at origin
        triangle1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(triangle1_nodes)

        # Triangle 2: Right triangle shifted
        triangle2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(triangle2_nodes)

        # Test all three operations - should not crash
        try:
            result_union = pad.boolean_union(polygon1, polygon2)
            self.assertIsInstance(result_union, self.PolygonPad)
        except NotImplementedError:
            pass

        try:
            result_intersection = pad.boolean_intersection(polygon1, polygon2)
            self.assertIsInstance(result_intersection, self.PolygonPad)
        except NotImplementedError:
            pass

        try:
            result_difference = pad.boolean_difference(polygon1, polygon2)
            self.assertIsInstance(result_difference, self.PolygonPad)
        except NotImplementedError:
            pass


class Polygon3dFp32TC(Polygon3dTB, unittest.TestCase):

    dtype = 'float32'
    Point = mm.Point3dFp32
    SegmentPad = mm.SegmentPadFp32
    CurvePad = mm.CurvePadFp32
    PolygonPad = mm.PolygonPadFp32
    SimpleArray = mm.SimpleArrayFloat32


class Polygon3dFp64TC(Polygon3dTB, unittest.TestCase):

    dtype = 'float64'
    Point = mm.Point3dFp64
    SegmentPad = mm.SegmentPadFp64
    CurvePad = mm.CurvePadFp64
    PolygonPad = mm.PolygonPadFp64
    SimpleArray = mm.SimpleArrayFloat64
