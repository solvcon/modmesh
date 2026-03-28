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

"""
Canvas utilities and curve/conic drawing utilities for pilot GUI.
"""

from .. import core, plot

from . import _gui_common

__all__ = [
    'Canvas',
    'BezierSample',
    'BezierSampler',
]


class Canvas(_gui_common.PilotFeature):
    """
    Canvas feature providing menu items for drawing curves and polygons.
    """

    def __init__(self, *args, **kw):
        super(Canvas, self).__init__(*args, **kw)
        self._world = core.WorldFp64()
        self._widget = None

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Create ICCAD-2013",
            tip="Create ICCAD-2013 polygon examples",
            func=self.mesh_iccad_2013,
        )

        tip = "Draw a sample S-shaped cubic Bezier curve with control points"
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Bezier S-curve",
            tip=tip,
            func=self._bezier_s_curve,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Bezier Arch",
            tip="Draw a sample arch-shaped cubic Bezier curve with control "
                "points",
            func=self._bezier_arch,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Bezier Loop",
            tip="Draw a sample loop-like cubic Bezier curve with control "
                "points",
            func=self._bezier_loop,
        )
        # TODO: Add more curve/conic samples in the next PRs,
        #       e.g. ellipse, parabola, hyperbola, etc.
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Ellipse",
            tip="Draw a sample ellipse (a=2, b=1)",
            func=self._ellipse,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Parabola",
            tip="Draw a sample parabola (y = 0.5*x^2)",
            func=self._parabola,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Hyperbola",
            tip="Draw a sample hyperbola (both branches)",
            func=self._hyperbola,
        )

    @staticmethod
    def _draw_layer(world, layer):
        point_type = core.Point3dFp64

        for polygon in layer.get_polys():
            segment_pad = core.SegmentPadFp64(ndim=2)

            for coord in polygon:
                segment_pad.append(core.Segment3dFp64(
                    point_type(coord[0][0], coord[0][1]),
                    point_type(coord[1][0], coord[1][1])
                ))

            world.add_segments(pad=segment_pad)

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

        self._draw_layer(self._world, layer)
        self._update_widget()

    def _update_widget(self):
        if self._widget is None:
            self._widget = self._mgr.add3DWidget()
        self._widget.updateWorld(self._world)
        self._widget.showMark()

    def _bezier_s_curve(self):
        bezier_sample = BezierSample.s_curve()
        sampler = BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _bezier_arch(self):
        bezier_sample = BezierSample.arch()
        sampler = BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _bezier_loop(self):
        bezier_sample = BezierSample.loop()
        sampler = BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _ellipse(self):
        # TODO: Make it in the next PR.
        raise NotImplementedError("Ellipse sample is not implemented yet")

    def _parabola(self):
        # TODO: Make it in the next PR.
        raise NotImplementedError("Parabola sample is not implemented yet")

    def _hyperbola(self):
        # TODO: Make it in the next PR.
        raise NotImplementedError("Hyperbola sample is not implemented yet")


class BezierSample(object):
    def __init__(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    @classmethod
    def s_curve(cls):
        return cls(p0=(0.0, 0.0), p1=(1.0, 3.0),
                   p2=(4.0, -1.0), p3=(5.0, 2.0))

    @classmethod
    def arch(cls):
        return cls(p0=(0.0, 0.0), p1=(1.5, 4.0),
                   p2=(3.5, 4.0), p3=(5.0, 0.0))

    @classmethod
    def loop(cls):
        return cls(p0=(0.0, 0.0), p1=(5.0, 3.0),
                   p2=(0.0, 3.0), p3=(5.0, 0.0))


class BezierSampler(object):
    def __init__(self, world, bezier_sample):
        self.world = world
        self.bezier_sample = bezier_sample

    def draw(self, nsample=50, fac=1.0, off_x=0.0, off_y=0.0,
             show_control_polygon=True, show_control_points=True):
        point_type = core.Point3dFp64
        bezier_sample = self.bezier_sample

        def _point(xy_pair):
            return point_type(xy_pair[0] * fac + off_x,
                              xy_pair[1] * fac + off_y, 0)

        p0 = _point(bezier_sample.p0)
        p1 = _point(bezier_sample.p1)
        p2 = _point(bezier_sample.p2)
        p3 = _point(bezier_sample.p3)

        bezier = self.world.add_bezier(p0=p0, p1=p1, p2=p2, p3=p3)
        bezier.sample(nsample)

        if show_control_polygon:
            self.world.add_segment(p0, p1)
            self.world.add_segment(p1, p2)
            self.world.add_segment(p2, p3)

        if show_control_points:
            mark_size = 0.1 * fac
            for point in (p0, p1, p2, p3):
                self.world.add_segment(
                    point_type(point.x - mark_size, point.y, 0),
                    point_type(point.x + mark_size, point.y, 0)
                )
                self.world.add_segment(
                    point_type(point.x, point.y - mark_size, 0),
                    point_type(point.x, point.y + mark_size, 0)
                )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
