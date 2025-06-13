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
from ..plot import svg
from ._gui_common import PilotFeature

__all__ = [  # noqa: F822
    'SVGFileDialog',
]


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
        svgParser = svg.PathParser(filename)
        svgParser.parse()
        epaths = svgParser.get_epaths()
        sp2d = []
        cp2d = []
        for p in epaths:
            closedp = p.get_closed_paths()
            sp2d.append(closedp[0])
            cp2d.append(closedp[1])

        world = core.WorldFp64()
        Point = core.Point3dFp64

        for i in range(len(sp2d)):
            for j in range(len(sp2d[i])):
                # the points are reflected to x-axis: (x, y) -> (x, -y)
                world.add_segment(Point(sp2d[i].x0_at(j), -sp2d[i].y0_at(j)),
                                  Point(sp2d[i].x1_at(j), -sp2d[i].y1_at(j)))

        for i in range(len(cp2d)):
            for j in range(len(cp2d[i])):
                # the points are reflected to x-axis: (x, y) -> (x, -y)
                b = world.add_bezier(p0=Point(cp2d[i].x0_at(j),
                                              -cp2d[i].y0_at(j),
                                              0),
                                     p1=Point(cp2d[i].x1_at(j),
                                              -cp2d[i].y1_at(j),
                                              0),
                                     p2=Point(cp2d[i].x2_at(j),
                                              -cp2d[i].y2_at(j),
                                              0),
                                     p3=Point(cp2d[i].x3_at(j),
                                              -cp2d[i].y3_at(j),
                                              0))
                b.sample(nlocus=5)

        wid = self._mgr.add3DWidget()
        wid.updateWorld(world)
        wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
