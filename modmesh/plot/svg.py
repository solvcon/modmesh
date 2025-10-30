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
Input, output, and process SVG (scalleable vector graphic).
"""

import re
import xml.etree.ElementTree as ET
from math import sin, cos, atan2, sqrt, radians, pi

import numpy as np

from .. import core

__all__ = [  # noqa: F822
    'SvgParser',
]


class SvgParser(object):
    """
    The SVG parser to extract SegmentPad and CurvePad from SVG file.

    Internally uses PathParser and ShapeParser to parse <path> and
    shape elements respectively.
    """
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def parse(self):
        path_parser = PathParser(file_path=self.file_path)
        path_parser.parse()
        self.spads = path_parser.spads
        self.cpads = path_parser.cpads

        shape_parser = ShapeParser(file_path=self.file_path)
        shape_parser.parse()
        self.spads.extend(shape_parser.spads)
        self.cpads.extend(shape_parser.cpads)

    def get_pads(self):
        return self.spads, self.cpads


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
        x_arc = (cx +
                 rx * np.cos(theta1 + t) * cos(phi) -
                 ry * np.sin(theta1 + t) * sin(phi))
        y_arc = (cy +
                 rx * np.cos(theta1 + t) * sin(phi) +
                 ry * np.sin(theta1 + t) * cos(phi))

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
                # Move position:
                #   command: [M|m] (relative position in lowercase command)
                #   parameter: (dx, dy)+
                dx, dy = coords[i:i+2]
                x_cur, y_cur = current_pos[0], current_pos[1]

                if cmd == 'M':
                    start_pos[0] = dx
                    start_pos[1] = dy
                else:
                    start_pos[0] = (x_cur + dx)
                    start_pos[1] = (y_cur + dy)

                current_pos = start_pos
                i += 2

                # implicit line-to command
                while i + 1 < len(coords):
                    dx, dy = coords[i:i+2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'm':
                        x_end = (x_cur + dx)
                        y_end = (y_cur + dy)

                    end = Point(x_end, y_end, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    i += 2

                last_control = None
            elif cmd in ('L', 'l'):
                # Draw lines from current position
                #   command: [L|l] (relative position in lowercase command)
                #   parameter: (dx1, dy1)+
                while i + 1 < len(coords):
                    dx, dy = coords[i:i+2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'l':
                        x_end = x_cur + dx
                        y_end = y_cur + dy

                    end = Point(x_end, y_end, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 2
            elif cmd in ('H', 'h'):
                # Draws horizontal lines
                #   command: [H|h] (relative position in lowercase command)
                #   parameter: (dx)+
                while i < len(coords):
                    dx = coords[i]

                    new_x = dx
                    old_y = current_pos[1]
                    if cmd == 'h':
                        new_x = current_pos[0] + dx

                    end = Point(new_x, old_y, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 1
            elif cmd in ('V', 'v'):
                # Draws a vertical line from current position
                #   command: [V|v] (relative position in lowercase command)
                #   parameter: (dy)+
                while i < len(coords):
                    dy = coords[i]
                    old_x = current_pos[0]

                    new_y = dy
                    if cmd == 'v':
                        new_y = current_pos[1] + dy

                    end = Point(old_x, new_y, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 1
            elif cmd in ('C', 'c'):
                # Draw cubic bezier curves
                #   command: [C|c] (relative position in lowercase command)
                #   parameter: (dx1, dy1, dx2, dy2, dx3, dy3)+
                while i + 5 < len(coords):
                    dx1, dy1, dx2, dy2, dx3, dy3 = coords[i:i+6]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'C':
                        x1 = dx1
                        y1 = dy1
                        x2 = dx2
                        y2 = dy2
                        x3 = dx3
                        y3 = dy3
                    else:
                        x1 = x_cur + dx1
                        y1 = y_cur + dy1
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2
                        x3 = x_cur + dx3
                        y3 = y_cur + dy3

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x3, y3, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 6
            elif cmd in ('S', 's'):
                # draw a smooth curves
                # command: [S|s] (relative position in lowercase command)
                # parameter: (dx2, dy2, dx3, dy3)+
                while i + 3 < len(coords):
                    dx2, dy2, dx3, dy3 = coords[i:i+4]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'S':
                        x2 = dx2
                        y2 = dy2
                        x3 = dx3
                        y3 = dy3
                    else:
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2
                        x3 = x_cur + dx3
                        y3 = y_cur + dy3

                    if (last_cmd in ('C', 'c', 'S', 's')
                            and last_control is not None):
                        x_lastc, y_lastc = last_control[0], last_control[1]
                        x1 = x_cur * 2 - x_lastc
                        y1 = y_cur * 2 - y_lastc
                    else:
                        x1, y1 = x_cur, y_cur

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x3, y3, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 4
            elif cmd in ('Q', 'q'):
                # Draw quadratic bezier curves
                #   command: [Q|q] (relative position in lowercase command)
                #   parameter: (dx1, dy1, dx2, dy2)+
                while i + 3 < len(coords):
                    dx1, dy1, dx2, dy2 = coords[i:i+4]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'Q':
                        x1 = dx1
                        y1 = dy1
                        x2 = dx2
                        y2 = dy2
                    else:
                        x1 = x_cur + dx1
                        y1 = y_cur + dy1
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x2, y2, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p1
                    i += 4
            elif cmd in ('T', 't'):
                # Draw a smooth quadratic Bezier curve from the current point
                # to the end point specified by `dx2 dy2`.
                #   command: [T|t] (relative position in lowercase command)
                #   parameter: (dx2, dy2)+
                while i + 1 < len(coords):
                    dx2, dy2 = coords[i:i+2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'T':
                        x2 = dx2
                        y2 = dy2
                    else:
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2

                    if (last_cmd in ('Q', 'q', 'T', 't')
                            and last_control is not None):
                        x_lastc, y_lastc = last_control[0], last_control[1]
                        x1 = x_cur * 2 - x_lastc
                        y1 = y_cur * 2 - y_lastc
                    else:
                        x1, y1 = x_cur, y_cur

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x2, y2, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p1
                    i += 2
            elif cmd in ('A', 'a'):
                # Draw a elliptical arc curves
                #   command: [A|a] (relative position in lowercase command)
                #   parameter: (rx, ry, angle,
                #               large-arc-flag, sweep-flag, dx, dy)+
                while i + 6 < len(coords):
                    start_pt = current_pos
                    (rx, ry, rotation,
                     Flarge_arc, Fsweep, dx, dy) = coords[i:i+7]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'a':
                        x_end = x_cur + dx
                        y_end = y_cur + dy

                    end = Point(x_end, y_end, 0)
                    arc_pts = self.calc_arc2pnts(start_pt, end, rx, ry,
                                                 rotation, Flarge_arc, Fsweep)

                    for p in range(arc_pts.shape[0]-1):
                        p_from = Point(arc_pts[p][0], arc_pts[p][1], 0)
                        p_to = Point(arc_pts[p+1][0], arc_pts[p+1][1], 0)
                        sp2d.append(Segment(p_from, p_to))

                    current_pos = end
                    i += 7
            elif cmd in ('Z', 'z'):
                # closed path:
                #   draw a straight line from the current position to
                #   the first point in the path.
                sp2d.append(Segment(current_pos, start_pos))
                current_pos = start_pos
                i += 1
            else:
                # [TODO] raise a value error
                i = len(coords)
            last_cmd = cmd
        return (sp2d, cp2d)

    def parse_dattr(self):
        d_attr = self.d_attr
        tokens = re.findall(r'([MLCSHVAZQTmlcshvazqt])|(-?\d*\.?\d+)', d_attr)

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


class PathParser(object):
    """
    The SVG <path> element parser to extract SegmentPad and CurvePad.

    Parse <path> elements from the SVG file and convert them into
    SegmentPad and CurvePad objects.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path
    """
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.epaths = []  # list of epath
        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def parse(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        pathElements = root.findall('.//svg:path', namespace)

        for elmnt in pathElements:
            d_attr = elmnt.attrib.get('d', '')
            fill_attr = elmnt.attrib.get('fill', '')
            epath = EPath(d_attr=d_attr, fill_attr=fill_attr)
            self.epaths.append(epath)
            # Get SegmentPad and CurvePad from the paths
            spad, cpad = epath.get_closed_paths()
            self.spads.append(spad)
            self.cpads.append(cpad)

    def get_epaths(self):
        return self.epaths


class EShapeBase(object):
    def __init__(self, fill_attr):
        self.fill_attr = fill_attr
        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def _calculate(self):
        raise NotImplementedError()

    def get_pads(self):
        return self.spads, self.cpads


class ECircle(EShapeBase):
    def __init__(self, cx, cy, r, fill_attr):
        super().__init__(fill_attr)
        self.cx = cx
        self.cy = cy
        self.r = r

        self._calculate()

    def _calculate(self):
        # Use 4 cubic Bezier curves to represent the circle
        # Magic constant for circular arc approximation:
        # kappa = 4/3 * tan(pi/8)
        # This value minimizes the radial error when approximating
        # a circular arc with a cubic Bezier curve
        kappa = 0.5522847498

        cpad = core.CurvePadFp64(ndim=2)

        # Top-right quadrant (0 to 90 degrees)
        p0 = core.Point3dFp64(self.cx + self.r, self.cy, 0)
        p1 = core.Point3dFp64(self.cx + self.r,
                              self.cy + self.r * kappa, 0)
        p2 = core.Point3dFp64(self.cx + self.r * kappa,
                              self.cy + self.r, 0)
        p3 = core.Point3dFp64(self.cx, self.cy + self.r, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Top-left quadrant (90 to 180 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy + self.r, 0)
        p1 = core.Point3dFp64(self.cx - self.r * kappa,
                              self.cy + self.r, 0)
        p2 = core.Point3dFp64(self.cx - self.r,
                              self.cy + self.r * kappa, 0)
        p3 = core.Point3dFp64(self.cx - self.r, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-left quadrant (180 to 270 degrees)
        p0 = core.Point3dFp64(self.cx - self.r, self.cy, 0)
        p1 = core.Point3dFp64(self.cx - self.r,
                              self.cy - self.r * kappa, 0)
        p2 = core.Point3dFp64(self.cx - self.r * kappa,
                              self.cy - self.r, 0)
        p3 = core.Point3dFp64(self.cx, self.cy - self.r, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-right quadrant (270 to 360 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy - self.r, 0)
        p1 = core.Point3dFp64(self.cx + self.r * kappa,
                              self.cy - self.r, 0)
        p2 = core.Point3dFp64(self.cx + self.r,
                              self.cy - self.r * kappa, 0)
        p3 = core.Point3dFp64(self.cx + self.r, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        self.cpads.append(cpad)


class EEllipse(EShapeBase):
    def __init__(self, cx, cy, rx, ry, fill_attr):
        super().__init__(fill_attr)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry

        self._calculate()

    def _calculate(self):
        cpad = core.CurvePadFp64(ndim=2)

        # Top-right quadrant (0 to 90 degrees)
        p0 = core.Point3dFp64(self.cx + self.rx, self.cy, 0)
        p1 = core.Point3dFp64(self.cx + self.rx,
                              self.cy + self.ry * 0.5522847498, 0)
        p2 = core.Point3dFp64(self.cx + self.rx * 0.5522847498,
                              self.cy + self.ry, 0)
        p3 = core.Point3dFp64(self.cx, self.cy + self.ry, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Top-left quadrant (90 to 180 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy + self.ry, 0)
        p1 = core.Point3dFp64(self.cx - self.rx * 0.5522847498,
                              self.cy + self.ry, 0)
        p2 = core.Point3dFp64(self.cx - self.rx,
                              self.cy + self.ry * 0.5522847498, 0)
        p3 = core.Point3dFp64(self.cx - self.rx, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-left quadrant (180 to 270 degrees)
        p0 = core.Point3dFp64(self.cx - self.rx, self.cy, 0)
        p1 = core.Point3dFp64(self.cx - self.rx,
                              self.cy - self.ry * 0.5522847498, 0)
        p2 = core.Point3dFp64(self.cx - self.rx * 0.5522847498,
                              self.cy - self.ry, 0)
        p3 = core.Point3dFp64(self.cx, self.cy - self.ry, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-right quadrant (270 to 360 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy - self.ry, 0)
        p1 = core.Point3dFp64(self.cx + self.rx * 0.5522847498,
                              self.cy - self.ry, 0)
        p2 = core.Point3dFp64(self.cx + self.rx,
                              self.cy - self.ry * 0.5522847498, 0)
        p3 = core.Point3dFp64(self.cx + self.rx, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        self.cpads.append(cpad)


class ERectangle(EShapeBase):
    def __init__(self, x, y, width, height, fill_attr):
        super().__init__(fill_attr)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self._calculate()

    def _calculate(self):
        p1 = core.Point3dFp64(self.x, self.y, 0)
        p2 = core.Point3dFp64(self.x + self.width, self.y, 0)
        p3 = core.Point3dFp64(self.x + self.width, self.y + self.height, 0)
        p4 = core.Point3dFp64(self.x, self.y + self.height, 0)

        spad = core.SegmentPadFp64(ndim=2)
        spad.append(core.Segment3dFp64(p1, p2))
        spad.append(core.Segment3dFp64(p2, p3))
        spad.append(core.Segment3dFp64(p3, p4))
        spad.append(core.Segment3dFp64(p4, p1))
        self.spads.append(spad)


class ELine(EShapeBase):
    def __init__(self, x1, y1, x2, y2, fill_attr):
        super().__init__(fill_attr)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self._calculate()

    def _calculate(self):
        p1 = core.Point3dFp64(self.x1, self.y1, 0)
        p2 = core.Point3dFp64(self.x2, self.y2, 0)

        spad = core.SegmentPadFp64(ndim=2)
        spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)


class EPolyline(EShapeBase):
    def __init__(self, points, fill_attr):
        super().__init__(fill_attr)
        self.points = points  # list of (x, y) tuples

        self._calculate()

    def _calculate(self):
        spad = core.SegmentPadFp64(ndim=2)
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            p1 = core.Point3dFp64(x1, y1, 0)
            p2 = core.Point3dFp64(x2, y2, 0)
            spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)


class EPolygon(EShapeBase):
    def __init__(self, points, fill_attr):
        super().__init__(fill_attr)
        self.points = points  # list of (x, y) tuples

        self._calculate()

    def _calculate(self):
        spad = core.SegmentPadFp64(ndim=2)
        num_points = len(self.points)
        for i in range(num_points):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % num_points]  # wrap around
            p1 = core.Point3dFp64(x1, y1, 0)
            p2 = core.Point3dFp64(x2, y2, 0)
            spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)


class ShapeParser(object):
    """
    Parse basic shapes from an SVG file to extract SegmentPad and CurvePad.

    Parses the basic shapes, including <circle>, <rect>, <ellipse>,
    <line>, <polyline>, and <polygon>, but excludes <path>.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorials/SVG_from_scratch/Basic_shapes
    """  # noqa: E501

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def parse(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}

        circleElements = root.findall('.//svg:circle', namespace)
        rectElements = root.findall('.//svg:rect', namespace)
        ellipseElements = root.findall('.//svg:ellipse', namespace)
        lineElements = root.findall('.//svg:line', namespace)
        polylineElements = root.findall('.//svg:polyline', namespace)
        polygonElements = root.findall('.//svg:polygon', namespace)

        shapes = []
        for element in circleElements:
            circle = ECircle(
                cx=float(element.attrib.get('cx', '0')),
                cy=float(element.attrib.get('cy', '0')),
                r=float(element.attrib.get('r', '0')),
                fill_attr=element.attrib.get('fill', ''),
            )
            shapes.append(circle)

        for element in rectElements:
            rect = ERectangle(
                x=float(element.attrib.get('x', '0')),
                y=float(element.attrib.get('y', '0')),
                width=float(element.attrib.get('width', '0')),
                height=float(element.attrib.get('height', '0')),
                fill_attr=element.attrib.get('fill', ''),
            )
            shapes.append(rect)

        for element in ellipseElements:
            ellipse = EEllipse(
                cx=float(element.attrib.get('cx', '0')),
                cy=float(element.attrib.get('cy', '0')),
                rx=float(element.attrib.get('rx', '0')),
                ry=float(element.attrib.get('ry', '0')),
                fill_attr=element.attrib.get('fill', ''),
            )
            shapes.append(ellipse)

        for element in lineElements:
            line = ELine(
                x1=float(element.attrib.get('x1', '0')),
                y1=float(element.attrib.get('y1', '0')),
                x2=float(element.attrib.get('x2', '0')),
                y2=float(element.attrib.get('y2', '0')),
                fill_attr=element.attrib.get('fill', ''),
            )
            shapes.append(line)

        for element in polylineElements:
            points_attr = element.attrib.get('points', '')
            points = []

            # TODO: handle commas and spaces properly. Assume points are in format: "x1,y1 x2,y2 x3,y3 ..."  # noqa: E501
            for pair in points_attr.strip().split():
                x_str, y_str = pair.split(',')
                points.append((float(x_str), float(y_str)))

            polyline = EPolyline(points=points,
                                 fill_attr=element.attrib.get('fill', ''))
            shapes.append(polyline)

        for element in polygonElements:
            points_attr = element.attrib.get('points', '')
            points = []

            # TODO: handle commas and spaces properly. Assume points are in format: "x1,y1 x2,y2 x3,y3 ..."  # noqa: E501
            for pair in points_attr.strip().split():
                x_str, y_str = pair.split(',')
                points.append((float(x_str), float(y_str)))

            polygon = EPolygon(points=points,
                               fill_attr=element.attrib.get('fill', ''))
            shapes.append(polygon)

        for shape in shapes:
            spad, cpad = shape.get_pads()
            self.spads.extend(spad)
            self.cpads.extend(cpad)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
