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

    def _compute_total_area(self, result_pad):
        """Helper to compute total unsigned area of all polygons in a pad."""
        total = 0.0
        for i in range(result_pad.num_polygons):
            total += abs(result_pad.get_polygon(i).compute_signed_area())
        return total

    def _assert_all_ccw(self, result_pad):
        """Assert all result polygons have counter-clockwise winding."""
        for i in range(result_pad.num_polygons):
            area = result_pad.get_polygon(i).compute_signed_area()
            self.assertGreaterEqual(area, 0.0,
                                    f"Polygon {i} has negative area {area}"
                                    " (clockwise winding)")

    def test_boolean_union_simple(self):
        """Test polygon boolean union with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2), area = 4
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3), area = 4
        # Overlap region: (1,1) to (2,2), area = 1
        # Union area = 4 + 4 - 1 = 7
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [7.0], rtol=1e-6)

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

        # Second square: (1,1) to (3,3)
        # Intersection should be (1,1) to (2,2), area = 1
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [1.0], rtol=1e-6)

    def test_boolean_difference_simple(self):
        """Test polygon boolean difference with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2), area = 4
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3)
        # Difference (polygon1 - polygon2) = L-shaped region, area = 4 - 1 = 3
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_difference(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [3.0], rtol=1e-6)

    def test_boolean_union_non_overlapping(self):
        """Test polygon boolean union with two non-overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (1,1), area = 1
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (2,2) to (3,3), area = 1
        # No overlap, union area = 2
        square2_nodes = [
            self.Point(2.0, 2.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(2.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [2.0], rtol=1e-6)

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

        result = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertEqual(result.num_polygons, 0)

    def test_boolean_operations_triangle(self):
        """Test polygon boolean operations with triangular polygons."""
        pad = self.PolygonPad(ndim=2)

        # Triangle 1: (0,0)-(2,0)-(0,2), area = 2
        triangle1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(triangle1_nodes)

        # Triangle 2: (1,1)-(3,1)-(1,3), area = 2
        triangle2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(triangle2_nodes)

        # Intersection: the overlap region is x>=1, y>=1, x+y<=2.
        # Since x>=1 and y>=1 imply x+y>=2, the only solution is the single
        # point (1,1). The two triangles touch at exactly one point, so
        # the intersection area is 0.
        result_intersection = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result_intersection, self.PolygonPad)
        intersection_area = self._compute_total_area(result_intersection)
        self.assertAlmostEqual(intersection_area, 0.0, places=5)

        # Union area = 2 + 2 - 0 = 4
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result_union, self.PolygonPad)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [4.0], rtol=1e-6)

        # Difference area = 2 - 0 = 2
        result_difference = pad.boolean_difference(polygon1, polygon2)
        self.assertIsInstance(result_difference, self.PolygonPad)
        diff_area = self._compute_total_area(result_difference)
        self.assert_allclose([diff_area], [2.0], rtol=1e-6)

    def test_boolean_containment(self):
        """Test boolean operations when one polygon contains the other."""
        pad = self.PolygonPad(ndim=2)

        # Large square: (0,0) to (4,4), area = 16
        large_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(4.0, 0.0, 0.0),
            self.Point(4.0, 4.0, 0.0),
            self.Point(0.0, 4.0, 0.0)
        ]
        polygon1 = pad.add_polygon(large_nodes)

        # Small square: (1,1) to (3,3), area = 4
        small_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(small_nodes)

        # Union = large square, area = 16
        result_union = pad.boolean_union(polygon1, polygon2)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [16.0], rtol=1e-6)

        # Intersection = small square, area = 4
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assert_allclose([inter_area], [4.0], rtol=1e-6)

        # Difference = large - small = 12
        result_diff = pad.boolean_difference(polygon1, polygon2)
        diff_area = self._compute_total_area(result_diff)
        self.assert_allclose([diff_area], [12.0], rtol=1e-6)

    def test_boolean_identical_polygons(self):
        """Test boolean operations when both polygons are identical."""
        pad = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(3.0, 0.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(0.0, 3.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes)
        polygon2 = pad.add_polygon(nodes)

        # Union(P, P) = P, area = 9
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [9.0], rtol=1e-6)

        # Intersection(P, P) = P, area = 9
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_inter)], [9.0], rtol=1e-6)

        # Difference(P, P) = empty
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self.assertEqual(result_diff.num_polygons, 0)

    def test_boolean_shared_edge(self):
        """Test boolean operations with two squares sharing an edge."""
        pad = self.PolygonPad(ndim=2)

        # Square 1: (0,0)-(1,1)
        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        # Square 2: (1,0)-(2,1), shares edge at x=1
        nodes2 = [
            self.Point(1.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union = (0,0)-(2,1), area = 2
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [2.0], rtol=1e-6)

        # Intersection: touching at edge only, area = 0
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assertAlmostEqual(inter_area, 0.0, places=5)

        # Difference(A, B) = square 1, area = 1
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_diff)], [1.0], rtol=1e-6)

    def test_boolean_commutativity(self):
        """Test that union and intersection are commutative,
        and difference is non-commutative."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union(A, B) == Union(B, A)
        union_ab = self._compute_total_area(
            pad.boolean_union(polygon1, polygon2))
        union_ba = self._compute_total_area(
            pad.boolean_union(polygon2, polygon1))
        self.assert_allclose([union_ab], [union_ba], rtol=1e-6)

        # Intersection(A, B) == Intersection(B, A)
        inter_ab = self._compute_total_area(
            pad.boolean_intersection(polygon1, polygon2))
        inter_ba = self._compute_total_area(
            pad.boolean_intersection(polygon2, polygon1))
        self.assert_allclose([inter_ab], [inter_ba], rtol=1e-6)

        # Difference is non-commutative: A-B=3, B-A=3 in area but
        # they are different regions. Verify areas are as expected.
        diff_ab = self._compute_total_area(
            pad.boolean_difference(polygon1, polygon2))
        diff_ba = self._compute_total_area(
            pad.boolean_difference(polygon2, polygon1))
        self.assert_allclose([diff_ab], [3.0], rtol=1e-6)
        self.assert_allclose([diff_ba], [3.0], rtol=1e-6)

    def test_boolean_partial_vertical_overlap(self):
        """Test with rectangles that partially overlap in Y."""
        pad = self.PolygonPad(ndim=2)

        # Rectangle 1: (0,0)-(2,3), area = 6
        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 3.0, 0.0),
            self.Point(0.0, 3.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        # Rectangle 2: (1,1)-(3,2), area = 2
        # Overlap region: (1,1)-(2,2), area = 1
        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(1.0, 2.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union = 6 + 2 - 1 = 7
        result_union = pad.boolean_union(polygon1, polygon2)
        self._assert_all_ccw(result_union)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [7.0], rtol=1e-6)

        # Intersection = 1
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        self._assert_all_ccw(result_inter)
        self.assert_allclose(
            [self._compute_total_area(result_inter)], [1.0], rtol=1e-6)

        # Difference = 6 - 1 = 5
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self._assert_all_ccw(result_diff)
        self.assert_allclose(
            [self._compute_total_area(result_diff)], [5.0], rtol=1e-6)

    def test_boolean_concave_polygon(self):
        """Test boolean operations with a concave (L-shaped) polygon."""
        pad = self.PolygonPad(ndim=2)

        # L-shaped polygon (concave):
        # (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2), area = 3
        l_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(1.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(l_nodes)

        # Square overlapping the L: (0.5, 0.5)-(1.5, 1.5), area = 1
        sq_nodes = [
            self.Point(0.5, 0.5, 0.0),
            self.Point(1.5, 0.5, 0.0),
            self.Point(1.5, 1.5, 0.0),
            self.Point(0.5, 1.5, 0.0)
        ]
        polygon2 = pad.add_polygon(sq_nodes)

        # The square partially extends outside the L's notch:
        #   Band [0.5, 1.0]: full square width (0.5 to 1.5), height 0.5 -> 0.5
        #   Band [1.0, 1.5]: only x in [0.5, 1.0] (L's left column), -> 0.25
        # Intersection area = 0.75
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assert_allclose([inter_area], [0.75], rtol=1e-6)

        # Union = L + square - intersection = 3 + 1 - 0.75 = 3.25
        result_union = pad.boolean_union(polygon1, polygon2)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [3.25], rtol=1e-6)

        # Difference(L, square) = L - intersection = 3 - 0.75 = 2.25
        result_diff = pad.boolean_difference(polygon1, polygon2)
        diff_area = self._compute_total_area(result_diff)
        self.assert_allclose([diff_area], [2.25], rtol=1e-6)

    def test_boolean_result_ccw_winding(self):
        """Test that all result polygons from boolean operations have
        counter-clockwise winding (non-negative signed area)."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        self._assert_all_ccw(pad.boolean_union(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_intersection(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_difference(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_difference(polygon2, polygon1))


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
