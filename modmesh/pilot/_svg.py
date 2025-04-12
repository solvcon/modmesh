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
import numpy as np
import re
import xml.etree.ElementTree as ET
from math import sin, cos, atan2, sqrt, radians, pi
from PySide6 import QtCore, QtWidgets

from .. import core
from ._gui_common import PilotFeature

__all__ = [  # noqa: F822
    'SVGFileDialog',
]


class EPath(object):
    def __init__(self, d_attr, fill_attr):
        """
        :param closedPaths: list of closed paths in a <path>.
        """
        self.d_attr = d_attr
        self.fill_attr = fill_attr
        self.cmds = self.parse_dattr()
        self.closedPaths = self.calc_vertices()

    def calc_arc2pnts(self, start_pt, end_pt, rx, ry, phi_deg, large_arc,
                      sweep, steps=40):
        """
        Populate points for an arc curve.

        :param start_pt: coordinates of starting point.
        :param end_pt: coordinates of ending point.
        :param rx, ry: radius of the ellipse.
        :param phi_deg: rotation of the ellipse (in degree).
        :param large_arc: arc size.
        :type large_arc: boolean (1: larger, 0: smaller)
        :param sweep: arc direction.
        :type sweep: boolean (1: clockwise or 0: counterclockwise)
        :param steps: number of points used to approximate the arc.
        """

        x0, y0 = start_pt[0], start_pt[1]
        x1, y1 = end_pt[0], end_pt[1]

        # convert angle to radians
        phi = radians(phi_deg)

        # compute rotated midpoint (x1', y1')
        dx = (x0 - x1) / 2.0
        dy = (y0 - y1) / 2.0
        xp = cos(phi) * dx + sin(phi) * dy
        yp = -sin(phi) * dx + cos(phi) * dy

        # correct radii
        rx = abs(rx)
        ry = abs(ry)
        r_check = (xp**2) / (rx**2) + (yp**2) / (ry**2)
        if r_check > 1:
            rx *= sqrt(r_check)
            ry *= sqrt(r_check)

        # compute center (cx', cy')
        num = rx**2 * ry**2 - rx**2 * yp**2 - ry**2 * xp**2
        denom = rx**2 * yp**2 + ry**2 * xp**2
        factor = sqrt(max(0, num / denom))  # ensure non-negative

        if large_arc == sweep:
            factor = -factor

        cxp = factor * (rx * yp) / ry
        cyp = factor * -(ry * xp) / rx

        # compute (cx, cy)
        cx = cos(phi) * cxp - sin(phi) * cyp + (x0 + x1) / 2
        cy = sin(phi) * cxp + cos(phi) * cyp + (y0 + y1) / 2

        # compute angles (start, delta)
        def angle(u, v):
            dot = u[0]*v[0] + u[1]*v[1]
            det = u[0]*v[1] - u[1]*v[0]
            return atan2(det, dot)

        u = [(xp - cxp)/rx, (yp - cyp)/ry]
        v = [(-xp - cxp)/rx, (-yp - cyp)/ry]

        theta1 = angle([1, 0], u)
        delta_theta = angle(u, v)

        if not sweep and delta_theta > 0:
            delta_theta -= 2 * pi
        elif sweep and delta_theta < 0:
            delta_theta += 2 * pi

        # using parametric equations for the ellipse and the arc angles
        t = np.linspace(0, delta_theta, num=steps)
        x_arc = cx + (rx * np.cos(theta1 + t) * cos(phi)) - (ry * np.sin(theta1 + t) * sin(phi))
        y_arc = cy + (rx * np.cos(theta1 + t) * sin(phi)) + (ry * np.sin(theta1 + t) * cos(phi))

        return np.column_stack((x_arc, y_arc))

    def calc_vertices(self):
        """
        path commands for `d` attribute:
            https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d
        """
        Point = core.Point3dFp64
        Segment = core.Segment3dFp64
        sp2d = core.SegmentPadFp64(ndim=2)
        cp2d = core.CurvePadFp64(ndim=2)

        commands = self.cmds
        start_pos = Point(0, 0, 0)
        current_pos = start_pos
        last_control = None
        last_cmd = None

        for idx, (cmd, coords) in enumerate(commands):
            i = 0
            if cmd in ('M', 'm'):
                # move to (x, y):
                #   M x y
                #   m dx dy (relative position to current position)
                x, y = coords[i], coords[i+1]

                if cmd == 'M':
                    start_pos[0] = x
                    start_pos[1] = y
                else:  # cmd is 'm', relative position
                    start_pos[0] = (current_pos[0] + x)
                    start_pos[1] = (current_pos[1] + y)

                current_pos = start_pos
                i += 2
                last_control = None
            elif cmd in ('L', 'l'):
                # line to (x, y):
                #   L x y
                #   l dx dy (relative position to current position)
                x, y = coords[i], coords[i+1]

                if cmd == 'L':  # absolute position
                    x_end = x
                    y_end = y
                else:  # cmd is 'l', relative position
                    x_end = current_pos[0] + x
                    y_end = current_pos[1] + y

                end = Point(x_end, y_end, 0)
                current_pos = end

                # add a Segment to SegmentPad
                sp2d.append(Segment(current_pos, end))
                i += 2
                last_control = None
            elif cmd == 'C':
                # draw cubic bezier curve:
                #   C ctl_x1, ctl_y1, ctl_x2, ctl_y2, endpos_x, endpos_y
                x_1, y_1, x_2, y_2, x_end, y_end = coords[i:i+6]

                p0 = current_pos
                p1 = Point(x_1, y_1, 0)
                p2 = Point(x_2, y_2, 0)
                p3 = Point(x_end, y_end, 0)

                # add a Curve to CurvePad with 4 control points
                cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                current_pos = p3
                last_control = p2
                i += 6
            elif cmd == 'c':
                # draw cubic bezier curve with relative position:
                #   c ctl_dx1, ctl_dy1, ctl_dx2, ctl_dy2, endpos_dx, endpos_dy
                while i + 5 < len(coords):
                    dx1, dy1, dx2, dy2, dx, dy = coords[i:i+6]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    p0 = current_pos
                    p1 = Point((x_cur + dx1), (y_cur + dy1), 0)
                    p2 = Point((x_cur + dx2), (y_cur + dy2), 0)
                    p3 = Point((x_cur + dx), (y_cur + dy), 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 6
            elif cmd == 'S':
                # draw a smooth curve (a shortcut version for cubic bezier)
                x_2, y_2, x_end, y_end = coords[i:i+4]
                x_cur, y_cur = current_pos[0], current_pos[1]

                # add a Curve to CurvePad with 4 control points
                if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                    x_lastc, y_lastc = last_control[0], last_control[1]
                    x_1, y_1 = (x_cur * 2 - x_lastc), (y_cur * 2 - y_lastc)
                else:
                    x_1, y_1 = x_cur, y_cur

                p0 = current_pos
                p1 = Point(x_1, y_1, 0)
                p2 = Point(x_2, y_2, 0)
                p3 = Point(x_end, y_end, 0)
                cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                current_pos = p3
                last_control = p2
                i += 4
            elif cmd == 's':
                # draw a smooth curve with relative positions
                # add a Curve to CurvePad with 4 control points
                while i + 3 < len(coords):
                    dx2, dy2, dx, dy = coords[i:i+4]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                        x_lastc, y_lastc = last_control[0], last_control[1]
                        x_p1 = x_cur * 2 - x_lastc
                        y_p1 = y_cur * 2 - y_lastc
                    else:
                        x_p1, y_p1 = x_cur, y_cur

                    p0 = current_pos
                    p1 = Point(x_p1, y_p1, 0)
                    p2 = Point(x_cur + dx2, y_cur + dy2, 0)
                    p3 = Point(x_cur + dx, y_cur + dy, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 4
            elif cmd in ('A', 'a'):
                # draw a elliptical arc curve
                start_pt = current_pos
                rx, ry, rotation, Flarge_arc, Fsweep, x, y = coords[i:i+7]

                if cmd == 'A':  # absolute position
                    x_end = x
                    y_end = y
                else:  # cmd is 'a', relative position
                    x_end = current_pos[0] + x
                    y_end = current_pos[1] + y

                end_pt = Point(x_end, y_end, 0)

                arc_pts = self.calc_arc2pnts(start_pt, end_pt, rx, ry,
                                             rotation, Flarge_arc, Fsweep)
                for i in range(arc_pts.shape[0]-1):
                    p_from = Point(arc_pts[i][0], arc_pts[i][1], 0)
                    p_to = Point(arc_pts[i+1][0], arc_pts[i+1][1], 0)
                    sp2d.append(Segment(p_from, p_to))

                current_pos = end_pt
                i += 7
            elif cmd in ('H', 'h'):
                # draws a horizontal line
                # for 'H', coord[0] is absolute position
                # for 'h', coord[0] is relative position from current position

                new_x = coords[i] if cmd == 'H' else current_pos[0] + coords[i]
                old_y = current_pos[1]
                end = Point(new_x, old_y, 0)

                # add a Segment to SegmentPad
                sp2d.append(Segment(current_pos, end))

                current_pos = end
                i += 1
                last_control = None
            elif cmd in ('V', 'v'):
                # draws a vertical line from current position
                # for 'V', coord[0] is absolute position
                # for 'v', coord[0] is relative position from current position

                old_x = current_pos[0]
                new_y = coords[i] if cmd == 'V' else current_pos[1] + coords[i]
                end = Point(old_x, new_y, 0)

                # add a Segment to SegmentPad
                sp2d.append(Segment(current_pos, end))

                current_pos = end
                i += 1
                last_control = None
            elif cmd in ('Z', 'z'):
                # form a closed path,
                #   draw a straight line from the current position to
                #   the first point in the path.
                sp2d.append(Segment(current_pos, start_pos))
                current_pos = start_pos
                i += 1
            else:
                i = len(coords)
            last_cmd = cmd
        return (sp2d, cp2d)

    def parse_dattr(self):
        d_attr = self.d_attr
        tokens = re.findall(r'([MLCSHAZmlcshvaz])|(-?\d*\.?\d+)', d_attr)

        commands = []
        current_command = None
        current_coords = []

        for cmd, val in tokens:
            if cmd:
                if current_command:
                    commands.append((current_command, current_coords))
                current_command = cmd
                current_coords = []
            else:
                current_coords.append(float(val))

        if current_command:
            commands.append((current_command, current_coords))

        return commands

    def get_closed_paths(self):
        return self.closedPaths

    def get_cmds(self):
        return self.cmds


class _PathParser(object):
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.Epaths = None  # list of Epath

    def parse(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        pathElements = root.findall('.//svg:path', namespace)

        paths = []
        for elmnt in pathElements:
            d_attr = elmnt.attrib.get('d', '')
            fill_attr = elmnt.attrib.get('fill', '')
            paths.append(EPath(d_attr=d_attr, fill_attr=fill_attr))
        self.Epaths = paths

    def get_EPaths(self):
        return self.Epaths


class SVGFileDialog(PilotFeature):
    """
    An example svg file can be downloaded from: https://www.svgrepo.com/svg/530293/tree-2
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

    def _load_svg_file(self, file_name):
        svgParser = _PathParser(file_name)
        svgParser.parse()
        Epaths = svgParser.get_EPaths()

        sp2d = []
        cp2d = []
        for p in Epaths:
            closedp = p.get_closed_paths()  # tuple
            sp2d.append(closedp[0])
            cp2d.append(closedp[1])

        world = core.WorldFp64()

        for i in range(len(sp2d)):
            for s in sp2d[i]:
                world.add_segment(s)

        for i in range(len(cp2d)):
            for c in cp2d[i]:
                b = world.add_bezier(p0=c[0],
                                     p1=c[1],
                                     p2=c[2],
                                     p3=c[3])
                b.sample(nlocus=5)

        wid = self._mgr.add3DWidget()
        wid.updateWorld(world)
        wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
