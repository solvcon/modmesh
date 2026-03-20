# Copyright (c) 2025, Jenny Yen <jenny35006@gmail.com>
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


"""
Show a SVG (scalleable vector graphic)
"""

import os
from PySide6 import QtCore, QtWidgets

from .. import core
from .. import apputil
from ..plot import svg
from ._gui_common import PilotFeature

__all__ = [  # noqa: F822
    'SVGFileDialog',
]

# Colors (r, g, b) for polygon boolean visualization
_COLOR_P1 = (91, 158, 240)       # blue
_COLOR_P2 = (240, 85, 69)        # red
_COLOR_TRAP_P1 = (110, 198, 255)  # light blue
_COLOR_TRAP_P2 = (255, 138, 101)  # light red
_COLOR_RESULT = (61, 220, 132)    # green


class SVGFileDialog(PilotFeature):
    """
    Download an example svg from: https://www.svgrepo.com/svg/530293/tree-2
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setDirectory(self._get_initial_path())
        self._diag.setWindowTitle('Open SVG file')

    def run(self):
        self._diag.open(self, QtCore.SLOT('on_finished()'))

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.fileMenu,
            text="Open SVG file",
            tip="Open SVG file",
            func=self.run,
        )

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)
        self._load_svg_file(filename=filenames[0])

    @staticmethod
    def _get_initial_path():
        found = ''
        for dp in ('.', core.__file__):
            dp = os.path.dirname(os.path.abspath(dp))
            dp2 = os.path.dirname(dp)

            while dp != dp2:
                tp = os.path.join(dp, "tests", "data")
                fp = os.path.join(tp, "tree.svg")
                if os.path.exists(fp):
                    found = fp
                    break
                dp = dp2
                dp2 = os.path.dirname(dp)
            if found:
                break
        return found

    def _load_svg_file(self, filename):
        parser = svg.SvgParser(file_path=filename)
        parser.parse()
        spads, cpads = parser.get_pads()

        world = core.WorldFp64()

        for spad in spads:
            # Flip against the X axis for GUI coordinate system
            spad.mirror(axis='x')
            world.add_segments(pad=spad)

        for cpad in cpads:
            # Flip against the X axis for GUI coordinate system
            cpad.mirror(axis='x')
            world.add_beziers(pad=cpad)

        wid = self._mgr.add3DWidget()
        wid.updateWorld(world)
        wid.showMark()

        # Add the data objects to the appenv for command-line access.
        cae = apputil.get_current_appenv()
        cae.locals['parser'] = parser
        cae.locals['world'] = world
        cae.locals['widget'] = wid

        # If the SVG contains 2+ polygon elements, run boolean operations
        if len(parser.polygon_vertices) >= 2:
            self._visualize_boolean(parser.polygon_vertices, cae)

    def _visualize_boolean(self, polygon_vertices_list, cae):
        """Run trapezoidal decomposition and boolean operations on the first
        two SVG polygon elements, then display colored results in additional
        widgets.

        Creates four additional 3D widgets with colored segments:
        - Decomposition: P1 outline (blue) + P2 outline (red) + trapezoids
        - Union: outlines + result (green)
        - Intersection: outlines + result (green)
        - Difference: outlines + result (green)
        """
        pad = core.PolygonPadFp64(ndim=2)

        nodes1 = self._svg_vertices_to_nodes(polygon_vertices_list[0])
        nodes2 = self._svg_vertices_to_nodes(polygon_vertices_list[1])

        p1 = pad.add_polygon(nodes1)
        p2 = pad.add_polygon(nodes2)

        # Trapezoidal decomposition
        trap_range1 = pad.decompose_to_trapezoid(p1.polygon_id)
        trap_range2 = pad.decompose_to_trapezoid(p2.polygon_id)
        trap_pad = pad.decomposed_trapezoids()

        # Boolean operations
        union_pad = pad.boolean_union(p1, p2)
        intersection_pad = pad.boolean_intersection(p1, p2)
        difference_pad = pad.boolean_difference(p1, p2)

        # Build a point-only world for camera/bounding-box setup.
        # Using points (not segments) avoids default-colored lines that
        # would overlap the colored ones.
        bounds_world = self._make_bounds_world(p1, p2)

        # Pre-build colored segment worlds (reused across widgets)
        w_p1 = self._make_polygon_world(p1)
        w_p2 = self._make_polygon_world(p2)
        w_trap1 = self._make_trapezoid_world(
            trap_pad, trap_range1[0], trap_range1[1])
        w_trap2 = self._make_trapezoid_world(
            trap_pad, trap_range2[0], trap_range2[1])

        # Pre-build filled quad worlds (points, 4 per quad)
        f_trap1 = self._make_filled_quad_world(
            trap_pad, trap_range1[0], trap_range1[1])
        f_trap2 = self._make_filled_quad_world(
            trap_pad, trap_range2[0], trap_range2[1])

        # --- Widget: Decomposition (colored, filled) ---
        wid_decomp = self._mgr.add3DWidget()
        wid_decomp.updateWorld(bounds_world)
        wid_decomp.addFilledPolygons(f_trap1, *_COLOR_TRAP_P1)
        wid_decomp.addFilledPolygons(f_trap2, *_COLOR_TRAP_P2)
        wid_decomp.addColoredSegments(w_p1, *_COLOR_P1)
        wid_decomp.addColoredSegments(w_p2, *_COLOR_P2)
        wid_decomp.addColoredSegments(w_trap1, *_COLOR_TRAP_P1)
        wid_decomp.addColoredSegments(w_trap2, *_COLOR_TRAP_P2)
        wid_decomp.showMark()

        # --- Widgets: Boolean results (colored, filled) ---
        ops = [
            ("union", union_pad),
            ("intersection", intersection_pad),
            ("difference", difference_pad),
        ]
        for name, result_pad in ops:
            wid = self._mgr.add3DWidget()
            wid.updateWorld(bounds_world)
            f_result = self._make_filled_result_world(result_pad)
            wid.addFilledPolygons(f_result, *_COLOR_RESULT)
            wid.addColoredSegments(w_p1, *_COLOR_P1)
            wid.addColoredSegments(w_p2, *_COLOR_P2)
            w_result = self._make_result_world(result_pad)
            wid.addColoredSegments(w_result, *_COLOR_RESULT)
            wid.showMark()

        # Print summary to console
        area_p1 = abs(p1.compute_signed_area())
        area_p2 = abs(p2.compute_signed_area())
        u_area = sum(abs(union_pad.get_polygon(i).compute_signed_area())
                     for i in range(union_pad.num_polygons))
        i_area = sum(
            abs(intersection_pad.get_polygon(i).compute_signed_area())
            for i in range(intersection_pad.num_polygons))
        d_area = sum(
            abs(difference_pad.get_polygon(i).compute_signed_area())
            for i in range(difference_pad.num_polygons))

        print(f"[Boolean] P1 area={area_p1:.2f}, P2 area={area_p2:.2f}")
        print(f"[Boolean] Decomposition: "
              f"P1={trap_range1[1]-trap_range1[0]} trapezoids, "
              f"P2={trap_range2[1]-trap_range2[0]} trapezoids")
        print(f"[Boolean] Union: {union_pad.num_polygons} trapezoids, "
              f"area={u_area:.2f}")
        print(f"[Boolean] Intersection: "
              f"{intersection_pad.num_polygons} trapezoids, "
              f"area={i_area:.2f}")
        print(f"[Boolean] Difference: "
              f"{difference_pad.num_polygons} trapezoids, "
              f"area={d_area:.2f}")

        # Store in appenv for command-line access
        cae.locals['polygon_pad'] = pad
        cae.locals['p1'] = p1
        cae.locals['p2'] = p2
        cae.locals['trap_pad'] = trap_pad
        cae.locals['union_pad'] = union_pad
        cae.locals['intersection_pad'] = intersection_pad
        cae.locals['difference_pad'] = difference_pad

    @staticmethod
    def _svg_vertices_to_nodes(vertices):
        """Convert SVG polygon vertices to Point3d nodes with Y-flip,
        ensuring CCW winding order."""
        Point = core.Point3dFp64

        # Flip Y for GUI coordinate system
        flipped = [(x, -y) for x, y in vertices]

        # Compute signed area via shoelace formula
        area = 0.0
        n = len(flipped)
        for i in range(n):
            j = (i + 1) % n
            area += flipped[i][0] * flipped[j][1]
            area -= flipped[j][0] * flipped[i][1]

        # Reverse if CW (negative signed area) to ensure CCW
        if area < 0:
            flipped.reverse()

        return [Point(x, y, 0) for x, y in flipped]

    @staticmethod
    def _make_polygon_world(polygon):
        """Create a World containing a polygon outline as segments."""
        Point = core.Point3dFp64
        Segment = core.Segment3dFp64
        world = core.WorldFp64()
        spad = core.SegmentPadFp64(ndim=2)
        for i in range(polygon.nnode):
            n0 = polygon.get_node(i)
            n1 = polygon.get_node((i + 1) % polygon.nnode)
            spad.append(Segment(
                Point(n0.x, n0.y, 0),
                Point(n1.x, n1.y, 0)))
        world.add_segments(pad=spad)
        return world

    @staticmethod
    def _make_bounds_world(*polygons):
        """Create a World containing only vertex points (no segments) from the
        given polygons.  ``updateWorld`` uses these points to compute the
        bounding box and position the camera without creating any visible
        default-colored line segments."""
        Point = core.Point3dFp64
        world = core.WorldFp64()
        for poly in polygons:
            for i in range(poly.nnode):
                n = poly.get_node(i)
                world.add_point(Point(n.x, n.y, 0))
        return world

    @staticmethod
    def _make_trapezoid_world(trap_pad, begin, end):
        """Create a World containing trapezoid edges as segments."""
        Point = core.Point3dFp64
        Segment = core.Segment3dFp64
        world = core.WorldFp64()
        spad = core.SegmentPadFp64(ndim=2)
        for i in range(begin, end):
            p0 = Point(trap_pad.x0(i), trap_pad.y0(i), 0)
            p1 = Point(trap_pad.x1(i), trap_pad.y1(i), 0)
            p2 = Point(trap_pad.x2(i), trap_pad.y2(i), 0)
            p3 = Point(trap_pad.x3(i), trap_pad.y3(i), 0)
            spad.append(Segment(p0, p1))
            spad.append(Segment(p1, p2))
            spad.append(Segment(p2, p3))
            spad.append(Segment(p3, p0))
        world.add_segments(pad=spad)
        return world

    @staticmethod
    def _make_filled_quad_world(trap_pad, begin, end):
        """Create a World with 4 points per trapezoid for filled rendering.
        addFilledPolygons reads every 4 consecutive points as one quad."""
        Point = core.Point3dFp64
        world = core.WorldFp64()
        for i in range(begin, end):
            world.add_point(Point(trap_pad.x0(i), trap_pad.y0(i), 0))
            world.add_point(Point(trap_pad.x1(i), trap_pad.y1(i), 0))
            world.add_point(Point(trap_pad.x2(i), trap_pad.y2(i), 0))
            world.add_point(Point(trap_pad.x3(i), trap_pad.y3(i), 0))
        return world

    @staticmethod
    def _make_filled_result_world(result_pad):
        """Create a World with 4 points per result polygon for filled
        rendering."""
        Point = core.Point3dFp64
        world = core.WorldFp64()
        for i in range(result_pad.num_polygons):
            poly = result_pad.get_polygon(i)
            for j in range(poly.nnode):
                n = poly.get_node(j)
                world.add_point(Point(n.x, n.y, 0))
        return world

    @staticmethod
    def _make_result_world(result_pad):
        """Create a World containing boolean result polygon edges."""
        Point = core.Point3dFp64
        Segment = core.Segment3dFp64
        world = core.WorldFp64()
        spad = core.SegmentPadFp64(ndim=2)
        for i in range(result_pad.num_polygons):
            poly = result_pad.get_polygon(i)
            for j in range(poly.nnode):
                n0 = poly.get_node(j)
                n1 = poly.get_node((j + 1) % poly.nnode)
                spad.append(Segment(
                    Point(n0.x, n0.y, 0),
                    Point(n1.x, n1.y, 0)))
        world.add_segments(pad=spad)
        return world

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
