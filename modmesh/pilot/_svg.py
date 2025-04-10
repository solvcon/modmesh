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
from PySide6 import QtCore, QtWidgets

from .. import core
from ._gui_common import PilotFeature

__all__ = [  # noqa: F822
    'SVGFileDialog',
]


class EPath(object):
    def __init__(self, dAttr, fillAttr):
        """
        :param closedPaths: list of closed paths in a <path>.
            Each closed paths in <path> is represented by a tuple (SegmentPad, CurvePad).
        """
        self.dAttr = dAttr
        self.fillAttr = fillAttr
        self.cmds = self.dAttr_parser()
        self.closedPaths = self.cal_vertices()
        # self.points = self.cal_vertices()

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
        Point = core.Point3dFp64  # Point object
        Segment = core.Segment3dFp64  # Segment object
        sp2d = core.SegmentPadFp64(ndim=2)
        cp2d = core.CurvePadFp64(ndim=2)

        commands = self.cmds
        start_pos = [0, 0]  # absolute position to (0, 0)
        current_pos = start_pos
        last_control = None
        last_cmd = None
        # closedPaths = []
        # vertices = np.empty((0, 2))  # create an empty 2D array

        for idx, (cmd, coords) in enumerate(commands):
            print(f"{idx}: '{cmd}', len of coords: {len(coords)}, coords: {coords}")
            print(f"start_pos: ({start_pos[0]}, {start_pos[1]})")
            print(f"[before] current_pos: ({current_pos[0]}, {current_pos[1]})")
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

                p_from = current_pos
                p_to = Point(x_end, y_end, 0)
                current_pos = p_to

                p_from -= start_pos
                p_to -= start_pos

                # add a Segment to SegmentPad
                sp2d.append(Segment(p_from, p_to))

                # vertices = np.vstack([vertices, end])
                i += 2
                last_control = None
            elif cmd == 'C':
                # draw cubic bezier curve:
                #   C ctl_x1, ctl_y1, ctl_x2, ctl_y2, endpos_x, endpos_y
                x_1, y_1, x_2, y_2, x_end, y_end = coords[i:i+6]

                x_cur, y_cur = current_pos[0], current_pos[1]

                # translate position to -(startX, startY)
                x_p0, y_p0 = x_cur - start_pos[0], y_cur - start_pos[1]
                x_p1, y_p1 = x_1 - start_pos[0], y_1 - start_pos[1]
                x_p2, y_p2 = x_2 - start_pos[0], y_2 - start_pos[1]
                x_p3, y_p3 = x_end - start_pos[0], y_end - start_pos[1]

                p0 = Point(x_p0, y_p0, 0)
                p1 = Point(x_p1, y_p1, 0)
                p2 = Point(x_p2, y_p2, 0)
                p3 = Point(x_p3, y_p3, 0)

                print(f"p0=({p0[0]}, {p0[1]})")
                print(f"p1=({p1[0]}, {p1[1]})")
                print(f"p2=({p2[0]}, {p2[1]})")
                print(f"p3=({p3[0]}, {p3[1]})")

                # add a Curve to CurvePad with 4 control points
                cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)
                print(f"cmd: {cmd}, add a curve, total curves: {len(cp2d)}")

                current_pos = [x_end, y_end]
                last_control = [x_2, y_2]

                i += 6
            elif cmd == 'c':
                # draw cubic bezier curve with relative position:
                #   c ctl_dx1, ctl_dy1, ctl_dx2, ctl_dy2, endpos_dx, endpos_dy
                while i + 5 < len(coords):
                    dx1, dy1, dx2, dy2, dx, dy = coords[i:i+6]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    # translate positions to -(startX, startY)
                    x_p0, y_p0 = x_cur - start_pos[0], y_cur - start_pos[1]
                    x_p1, y_p1 = (x_cur + dx1) - start_pos[0], (y_cur + dy1) - start_pos[1]
                    x_p2, y_p2 = (x_cur + dx2) - start_pos[0], (y_cur + dy2) - start_pos[1]
                    x_p3, y_p3 = (x_cur + dx) - start_pos[0], (y_cur + dy) - start_pos[1]

                    p0 = Point(x_p0, y_p0, 0)
                    p1 = Point(x_p1, y_p1, 0)
                    p2 = Point(x_p2, y_p2, 0)
                    p3 = Point(x_p3, y_p3, 0)

                    print(f"p0=({p0[0]}, {p0[1]})")
                    print(f"p1=({p1[0]}, {p1[1]})")
                    print(f"p2=({p2[0]}, {p2[1]})")
                    print(f"p3=({p3[0]}, {p3[1]})")

                    # add a Curve to CurvePad with 4 control points
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)
                    print(f"cmd: {cmd}, add a curve, total curves: {len(cp2d)}")

                    current_pos = [(x_cur + dx), (y_cur + dy)]
                    last_control = [(x_cur + dx2), (y_cur + dy2)]
                    i += 6
            elif cmd == 'S':
                # draw a smooth curve (a shortcut version for cubic bezier)
                x_2, y_2, x_end, y_end = coords[i:i+4]
                x_cur, y_cur = current_pos[0], current_pos[1]
                x_lastc, y_lastc = last_control[0], last_control[1]

                # add a Curve to CurvePad with 4 control points
                if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                    x_p1, y_p1 = (x_cur * 2 - x_lastc) - start_pos[0], (y_cur * 2 - y_lastc) - start_pos[1]
                else:
                    x_p1, y_p1 = x_cur - start_pos[0], y_cur - start_pos[1]

                # translate positions to -(startX, startY)
                x_p0, y_p0 = x_cur - start_pos[0], y_cur - start_pos[1]
                x_p2, y_p2 = x_2 - start_pos[0], y_2 - start_pos[1]
                x_p3, y_p3 = x_end - start_pos[0], y_end - start_pos[1]

                p0 = Point(x_p0, y_p0, 0)
                p1 = Point(x_p1, y_p1, 0)
                p2 = Point(x_p2, y_p2, 0)
                p3 = Point(x_p3, y_p3, 0)

                print(f"p0=({p0[0]}, {p0[1]})")
                print(f"p1=({p1[0]}, {p1[1]})")
                print(f"p2=({p2[0]}, {p2[1]})")
                print(f"p3=({p3[0]}, {p3[1]})")

                cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)
                print(f"cmd: {cmd}, add a curve, total curves: {len(cp2d)}")

                current_pos = [x_end, y_end]
                last_control = [x_2, y_2]
                i += 4
            elif cmd == 's':
                # draw a smooth curve with relative positions
                # add a Curve to CurvePad with 4 control points
                while i + 3 < len(coords):
                    dx2, dy2, dx, dy = coords[i:i+4]
                    x_cur, y_cur = current_pos[0], current_pos[1]
                    x_lastc, y_lastc = last_control[0], last_control[1]

                    if last_cmd in ('C', 'c', 'S', 's') and last_control is not None:
                        x_p1, y_p1 = (x_cur * 2 - x_lastc) - start_pos[0], (y_cur * 2 - y_lastc) - start_pos[1]
                    else:
                        x_p1, y_p1 = x_cur - start_pos[0], y_cur - start_pos[1]

                    # translate positions to -(startX, startY)
                    x_p0, y_p0 = x_cur - start_pos[0], y_cur - start_pos[1]
                    x_p2, y_p2 = (x_cur + dx2) - start_pos[0], (y_cur + dy2) - start_pos[1]
                    x_p3, y_p3 = (x_cur + dx) - start_pos[0], (y_cur + dy) - start_pos[1]

                    p0 = Point(x_p0, y_p0, 0)
                    p1 = Point(x_p1, y_p1, 0)
                    p2 = Point(x_p2, y_p2, 0)
                    p3 = Point(x_p3, y_p3, 0)

                    print(f"p0=({p0[0]}, {p0[1]})")
                    print(f"p1=({p1[0]}, {p1[1]})")
                    print(f"p2=({p2[0]}, {p2[1]})")
                    print(f"p3=({p3[0]}, {p3[1]})")

                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)
                    print(f"cmd: {cmd}, add a curve, total curves: {len(cp2d)}")

                    current_pos = [(x_cur + dx), (y_cur + dy)]
                    last_control = [(x_cur + dx2), (y_cur + dy2)]

                    i += 4
            elif cmd == 'A':
                # draw a elliptical arc curve
                start_pt = current_pos
                rx, ry, rotation, Flarge_arc, Fsweep, x, y = coords[i:i+7]

                if cmd == 'M':  # absolute position
                    x_end = x
                    y_end = y
                else:  # cmd is 'a', relative position
                    x_end = current_pos[0] + x
                    y_end = current_pos[1] + y

                end_pt = Point(x_end, y_end, 0)

                # [TODO] approximate a elliptical arc curve by one cbc
                p0, p1, p2, p3 = self.cal_arc2pnts(start_pt,
                                                   end_pt,
                                                   rx, ry,
                                                   rotation,
                                                   Flarge_arc, Fsweep)

                # [TODO] add a Curve to CurvePad with 4 control points
                cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                # arc_pts = self.cal_arc2pnts(start_pt, end_pt, rx, ry, rotation,
                #                            Flarge_arc, Fsweep)
                # vertices = np.vstack([vertices, arc_pts])
                current_pos = end_pt
                i += 7
            elif cmd in ('H', 'h'):
                # draws a horizontal line
                # for 'H', coord[0] is absolute position
                # for 'h', coord[0] is relative position from current position

                new_x = coords[i] if cmd == 'H' else current_pos[0] + coords[i]  # dx
                old_y = current_pos[1]
                end = Point(new_x, old_y, 0)

                # add a Segment to SegmentPad
                sp2d.append(Segment(current_pos, end))

                # vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd in ('V', 'v'):
                # draws a vertical line from current position
                # for 'V', coord[0] is absolute position
                # for 'v', coord[0] is relative position from current position

                old_x = current_pos[0]
                new_y = coords[i] if cmd == 'V' else current_pos[1] + coords[i]  # dy
                end = Point(old_x, new_y, 0)
               
                # add a Segment to SegmentPad
                sp2d.append(Segment(current_pos, end))

                # vertices = np.vstack([vertices, end])
                current_pos = end
                i += 1
                last_control = None
            elif cmd in ('Z', 'z'):
                # form a closed path,
                #   draw a straight line from the current position to the first point in the path.

                x_p0, y_p0 = current_pos[0] - start_pos[0], current_pos[1] - start_pos[1]
                x_p1, y_p1 = start_pos[0] - start_pos[0], start_pos[1] - start_pos[1]

                p_from = Point(x_p0, y_p0, 0)
                p_to = Point(x_p1, y_p1, 0)

                print(f"p_from=({p_from[0]}, {p_from[1]})")
                print(f"p_to=({p_to[0]}, {p_to[1]})")

                # add a Segment to SegmentPad
                sp2d.append(Segment(p_from, p_to))
                print(f"cmd: {cmd}, add a segment, total segments: {len(sp2d)}")
                i += 1
            else:
                # [TODO] invalid path command, need to rasie an Error
                i = len(coords)
            last_cmd = cmd
            print(f"[after] start_pos: ({start_pos[0]}, {start_pos[1]})")
            print(f"[after] current_pos: ({current_pos[0]}, {current_pos[1]})")
            print('\n')
        # self.closedPaths = (sp2d, cp2d)
        return (sp2d, cp2d)

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

    def getCmds(self):
        return self.cmds


class _pathParser(object):
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.Epaths = None  # list of Epath

    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        pathElements = root.findall('.//svg:path', namespace)

        # print(len(total_paths))
        paths = []
        d_attr = pathElements[0].attrib.get('d', '')
        fill_attr = pathElements[0].attrib.get('fill', '')
        paths.append(EPath(dAttr=d_attr, fillAttr=fill_attr))

        # Test for <path>[0] which is without 'A'/'a' command
        """ for elmnt in pathElements:
            d_attr = elmnt.attrib.get('d', '')
            fill_attr = elmnt.attrib.get('fill', '')
            pdb.set_trace()
            paths.append(EPath(dAttr=d_attr, fillAttr=fill_attr))
        """
        self.Epaths = paths

    def getEPaths(self):
        return self.Epaths


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

        # get closed paths representing by SegmentPad and CurvePad
        # svgParser.getEPaths() will return all EPath (representing <path>). 
        # Each EPath has  a set of closed paths
        # Each closed paths in <path> is represented by a tuple (SegmentPad, CurvePad)
        Epaths = svgParser.getEPaths()

        for p in Epaths:
            cmds = p.getCmds()  # list of tuple
            closedp = p.getClosedPaths()  # tuple
            sp2d = closedp[0]
            cp2d = closedp[1]
            print(f"total commands: {len(cmds)}")
            print(f"number of segments: {len(sp2d)}")
            print(f"number of curves: {len(cp2d)}")

        # print(f"sp2d[0] type: {type(sp2d[0])}")  # <class '_modmesh.Segment3dFp64'>
        # print(f"cp2d[0] type: {type(cp2d[0])}")  # <class '_modmesh.Bezier3dFp64'>

        # sample curves to create segment pad
        # sp_cp2d = cp2d.sample(length=0.5)
        # print(f"number of segments from curves: {len(sp_cp2d)}")

        world = core.WorldFp64()

        for i in range(len(sp2d)):
            # print(f"point {i} in ({sp2d[i][0]}, {sp2d[i][1]})")
            world.add_segment(sp2d[i])

        for i in range(len(cp2d)):
            b = world.add_bezier(p0=cp2d[i][0],
                                 p1=cp2d[i][1],
                                 p2=cp2d[i][2],
                                 p3=cp2d[i][3])
            b.sample(nlocus=5)

        wcp = world.curves  # get world CurvePad
        print(f"world segments: {world.nsegment}")
        print(f"world curves: {len(wcp)}")
        print(f"wcp[0] type: {type(wcp[0])}")

        for c in range(len(wcp)):
            print(f"#{c} bezier")
            print(f"p0=({wcp.x0_at(c)}, {wcp.y0_at(c)})")
            print(f"p1=({wcp.x1_at(c)}, {wcp.y1_at(c)})")
            print(f"p2=({wcp.x2_at(c)}, {wcp.y2_at(c)})")
            print(f"p3=({wcp.x3_at(c)}, {wcp.y3_at(c)})")
            print("\n")

        wid = self._mgr.add3DWidget()
        wid.updateWorld(world)
        wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
