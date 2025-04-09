# Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from PySide6 import QtCore, QtWidgets

from .. import core
from ._gui_common import PilotFeature

__all__ = [  # noqa: F822
    'SVGFileDialog',
]


class EPath(object):
    def __init__(self, dAttr, fillAttr):
        self.dAttr = dAttr
        self.fillAttr = fillAttr
        self.closedPaths = None
        self.pathcmds = self.dAttr_parser()
        self.points = self.cal_vertices()

    def cal_cubic_bezier(self, p0, p1, p2, p3, num=20):
        """
        comput a cubic bezier curve
        """
        t = np.linspace(0, 1, num)
        return (
            (1 - t)[:, None]**3 * p0 +
            3 * (1 - t)[:, None]**2 * t[:, None] * p1 +
            3 * (1 - t)[:, None] * t[:, None]**2 * p2 +
            t[:, None]**3 * p3
        )

    def cal_arc2pnts(self, start_pt, end_pt, rx, ry, phi_deg, large_arc, sweep,
                     steps=40):
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
        x_arc = cx + rx * np.cos(theta1 + t) * cos(phi) - ry * np.sin(theta1 + t) * sin(phi)
        y_arc = cy + rx * np.cos(theta1 + t) * sin(phi) + ry * np.sin(theta1 + t) * cos(phi)

        return np.column_stack((x_arc, y_arc))

    def cal_vertices(self):
        """
        path commands for `d` attribute:
            https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d
        """
        commands = self.pathcmds
        vertices = np.empty((0, 2))  # create an empty 2D array
        current_pos = np.array([0.0, 0.0])
        last_control = None
        last_cmd = None
        closedPaths = []

        for cmd, coords in commands:
            i = 0
            if cmd in ('M', 'm'):
                # move to (x, y):
                #   M x y
                #   m dx dy (relative position to current position)
                x, y = coords[i], coords[i+1]
                current_pos = np.array([x, y]) if cmd == 'M' else current_pos + np.array([x, y])
                vertices = np.vstack([vertices, current_pos])
                i += 2
                last_control = None
            elif cmd in ('L', 'l'):
                # line to (x, y):
                #   L x y
                #   l dx dy (relative position to current position)
                x, y = coords[i], coords[i+1]
                end = np.array([x, y]) if cmd == 'L' else current_pos + np.array([x, y])
                vertices = np.vstack([vertices, end])
                current_pos = end
                i += 2
                last_control = None
            elif cmd == 'C':
                # draw cubic bezier curve:
                #   C ctl_x1, ctl_y1, ctl_x2, ctl_y2, endpos_x, endpos_y
                x1, y1, x2, y2, x, y = coords[i:i+6]
                p1 = np.array([x1, y2])
                p2 = np.array([x2, y2])
                p3 = np.array([x, y])

                points = self.cal_cubic_bezier(current_pos, p1, p2, p3)
                vertices = np.vstack([vertices, points])
                current_pos = p3
                last_control = p2
                i += 6
            elif cmd == 'c':
                # draw cubic bezier curve with relative position:
                #   c ctl_dx1, ctl_dy1, ctl_dx2, ctl_dy2, endpos_dx, endpos_dy
                while i + 5 < len(coords):
                    dx1, dy1, dx2, dy2, dx, dy = coords[i:i+6]
                    p1 = current_pos + np.array([dx1, dx2])
                    p2 = current_pos + np.array([dx2, dy2])
                    p3 = current_pos + np.array([dx, dy])
                    
                    points = self.cal_cubic_bezier(current_pos, p1, p2, p3)
                    vertices = np.vstack([vertices, points])
                    current_pos = p3
                    last_control = p2
                    i += 6
            elif cmd == 'S':
                # draw a smooth curve (a shortcut version for cubic bezier)
                x2, y2, x, y = coords[i:i+4]

                if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                    p1 = current_pos * 2 - last_control
                else:
                    p1 = current_pos

                p2 = np.array([x2, y2])
                p3 = np.array([x, y])

                points = self.cal_cubic_bezier(current_pos, p1, p2, p3)
                vertices = np.vstack([vertices, points])
                current_pos = p3
                last_control = p2
                i += 4
            elif cmd == 's':
                # draw a smooth curve with relative positions
                while i + 3 < len(coords):
                    dx2, dy2, dx, dy = coords[i:i+4]
                    if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                        p1 = current_pos * 2 - last_control
                    else:
                        p1 = current_pos

                    p2 = current_pos + np.array([dx2, dy2])
                    p3 = current_pos + np.array([dx, dy])

                    points = self.cal_cubic_bezier(current_pos, p1, p2, p3)
                    vertices = np.vstack([vertices, points])
                    current_pos = p3
                    last_control = p2
                    i += 4
            elif cmd == 'A':
                # draw a arc curve
                start_pt = current_pos
                rx, ry, rotation, Flarge_arc, Fsweep, x1, y1 = coords[i:i+7]
                end_pt = np.array([x1, y1])

                arc_pts = self.cal_arc2pnts(start_pt, end_pt, rx, ry, rotation,
                                            Flarge_arc, Fsweep)

                vertices = np.vstack([vertices, arc_pts])
                current_pos = end_pt
                i += 7
            elif cmd == 'a':
                # draw a arc curve with relative position
                start_pt = current_pos
                rx, ry, rotation, Flarge_arc, Fsweep, dx, dy = coords[i:i+7]
                end_pt = current_pos + np.array([dx, dy])

                arc_pts = self.cal_arc2pnts(start_pt, end_pt, rx, ry, rotation,
                                            Flarge_arc, Fsweep)

                vertices = np.vstack([vertices, arc_pts])
                current_pos = end_pt
                i += 7
            elif cmd == 'H':
                # draws a horizontal line
                x = coords[i]
                new_x, old_y = x, current_pos[1]
                end = np.array([new_x, old_y])
                vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd == 'h':
                # draws a horizontal line with delta_x
                dx = coords[i]
                new_x, old_y = current_pos[0] + dx, current_pos[1]
                end = np.array([new_x, old_y])
                vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd == 'V':
                # draws a vertical line
                y = coords[i]
                old_x, new_y = current_pos[0], current_pos[1] + dy
                end = np.array([old_x, new_y])
                vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd == 'v':
                # draws a horizontal line with delta_x
                dy = coords[i]
                old_x, new_y = current_pos[0], current_pos[1] + dy
                end = np.array([old_x, new_y])
                vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd in ['Z', 'z']:
                # form a closed path, push the vertices back to the start
                vertices = np.vstack([vertices, vertices[0]])
                closedPaths.append(vertices)
                vertices = np.empty((0, 2))
                i += 1
            else:
                i = len(coords)
            last_cmd = cmd
        self.closedPaths = closedPaths
        return vertices

    def dAttr_parser(self):
        d_attr = self.dAttr
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

    def getClosedPaths(self):
        return self.closedPaths


class _pathParser(object):
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.paths = None  # list of Path

    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}

        paths = []
        for elmnt in root.findall('.//svg:path', namespace):
            d_attr = elmnt.attrib.get('d', '')
            fill_attr = elmnt.attrib.get('fill', '')
            paths.append(EPath(dAttr=d_attr, fillAttr=fill_attr))
        self.paths = paths

    def getEPaths(self):
        return self.paths


def _plot_paths(paths):
    # [TODO] use Qt to plot paths
    fig, ax = plt.subplots()
    lc = LineCollection(paths, linewidths=1.5, linestyle='solid')
    ax.add_collection(lc)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.invert_yaxis()
    plt.grid(True)
    plt.show()


class SVGFileDialog(PilotFeature):
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
        svgParser = _pathParser(filename)
        svgParser.parse()
        paths = svgParser.getEPaths()

        cpaths = []
        all_pts = []
        for p in paths:
            closedPaths = p.getClosedPaths()
            for cp in closedPaths:
                all_pts.append(cp)
                cpaths.append(cp.tolist())

        all_pts = np.vstack(all_pts)
        # [TODO] Prepare the CurvePad and SegementPad for ploting
        _plot_paths(cpaths)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
