# Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
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

from .. import core, plot

from ._gui_common import PilotFeature


class Canvas(PilotFeature):
    """
    Create canvas windows for render layer.
    """

    def __init__(self):
        self.world = core.WorldFp64()

    def draw_layer(self, layer):
        P = core.Point3dFp64

        for poly in layer.get_polys():
            spad = core.SegmentPadFp64(ndim=2)

            for coord in poly:
                spad.append(core.Segment3dFp64(
                    P(coord[0][0], coord[0][1]),
                    P(coord[1][0], coord[1][1])))

            self.world.add_segments(pad=spad)

    def get_world(self):
        return self.world


class CanvasMenu(PilotFeature):
    """
    Create sample canvas windows.
    """

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Create ICCAD-2013",
            tip="Create ICCAD-2013 polygon examples",
            func=self.mesh_iccad_2013,
        )

    def mesh_iccad_2013(self):
        layer = plot.plane_layer.PlaneLayer()
        layer.add_figure("RECT N M1 70 800 180 40")
        layer.add_figure(
            "PGON N M1 70 720 410 720 410 920 70 920 "
            "70 880 370 880 370 760 70 760"
        )
        layer.add_figure("RECT N M1 70 1060 180 40")
        layer.add_figure(
            "PGON N M1 70 980 410 980 410 1180 70 1180 "
            "70 1140 370 1140 370 1020 70 1020"
        )

        canvas = Canvas()
        canvas.draw_layer(layer)

        wid = self._mgr.add3DWidget()
        wid.updateWorld(canvas.get_world())
        wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
