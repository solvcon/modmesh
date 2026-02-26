# Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
# Copyright (c) 2026, Anchi Liu <phy.tiger@gmail.com>
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

import numpy as np

from .. import core, plot

from ._gui_common import PilotFeature

__all__ = [
    'Canvas',
    'Ellipse',
    'EllipseSampler',
    'Parabola',
    'ParabolaSampler',
    'Hyperbola',
    'HyperbolaSampler',
    'BezierSample',
    'BezierSampler',
]


def _populate_sampler_points(curve, npoint=100, fac=1.0, off_x=0.0,
                             off_y=0.0):
    if npoint < 1:
        raise ValueError("npoint must be at least 1")

    points = curve.calc_points(npoint)
    points.x.ndarray[:] *= fac
    points.y.ndarray[:] *= fac
    points.x.ndarray[:] += off_x
    points.y.ndarray[:] += off_y
    return points


def _draw_piecewise_bezier(world, points, spacing=0.01):
    if points is None:
        raise RuntimeError(
            "populate_points() must be called before draw_cbc()"
        )
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if len(points) < 2:
        return

    point_type = core.Point3dFp64
    point_x = points.x.ndarray
    point_y = points.y.ndarray
    segment_length = np.hypot(
        point_x[:-1] - point_x[1:],
        point_y[:-1] - point_y[1:],
    )
    nsample = np.maximum((segment_length // spacing).astype(int) - 1, 2)

    for index in range(len(points) - 1):
        p0 = np.array(points[index])
        p3 = np.array(points[index + 1])
        p1 = p0 + (1.0 / 3.0) * (p3 - p0)
        p2 = p0 + (2.0 / 3.0) * (p3 - p0)
        bezier = world.add_bezier(
            p0=point_type(p0[0], p0[1], 0.0),
            p1=point_type(p1[0], p1[1], 0.0),
            p2=point_type(p2[0], p2[1], 0.0),
            p3=point_type(p3[0], p3[1], 0.0),
        )
        bezier.sample(int(nsample[index]))


class Canvas(PilotFeature):
    """
    Canvas feature providing menu items for drawing curves and polygons.
    """

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
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Ellipse",
            tip="Draw a sample ellipse (a=2, b=1)",
            func=self._ellipse_window,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Parabola",
            tip="Draw a sample parabola (y = 0.5*x^2)",
            func=self._parabola_window,
        )
        self._add_menu_item(
            menu=self._mgr.canvasMenu,
            text="Sample: Hyperbola",
            tip="Draw a sample hyperbola (both branches)",
            func=self._hyperbola_window,
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

        world = core.WorldFp64()
        self._draw_layer(world, layer)
        self._show_world(world)

    def _show_world(self, world):
        widget = self._mgr.add3DWidget()
        widget.updateWorld(world)
        widget.showMark()

    def _bezier_s_curve(self):
        world = core.WorldFp64()
        bezier_sample = BezierSample.s_curve()
        sampler = BezierSampler(world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._show_world(world)

    def _bezier_arch(self):
        world = core.WorldFp64()
        bezier_sample = BezierSample.arch()
        sampler = BezierSampler(world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._show_world(world)

    def _bezier_loop(self):
        world = core.WorldFp64()
        bezier_sample = BezierSample.loop()
        sampler = BezierSampler(world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._show_world(world)

    def _ellipse_window(self):
        world = core.WorldFp64()
        ellipse = Ellipse(a=2.0, b=1.0)
        sampler = EllipseSampler(world, ellipse)
        sampler.populate_points(npoint=100, fac=1.0, off_x=0.0, off_y=0.0)
        sampler.draw_cbc()
        self._show_world(world)

    def _parabola_window(self):
        world = core.WorldFp64()
        parabola = Parabola(a=0.5, t_min=-3.0, t_max=6.0)
        sampler = ParabolaSampler(world, parabola)
        sampler.populate_points(npoint=100, fac=1.0, off_x=0.0, off_y=0.0)
        sampler.draw_cbc()
        self._show_world(world)

    def _hyperbola_window(self):
        world = core.WorldFp64()
        hyperbola = Hyperbola(a=1.0, b=1.0, t_min=-2.0, t_max=2.0)

        right_sampler = HyperbolaSampler(world, hyperbola)
        right_sampler.populate_points(
            npoint=100,
            fac=1.0,
            off_x=0.0,
            off_y=0.0,
        )
        right_sampler.draw_cbc()

        left_sampler = HyperbolaSampler(world, hyperbola)
        left_sampler.populate_points(
            npoint=100,
            fac=1.0,
            off_x=0.0,
            off_y=0.0,
        )
        left_sampler.points.x.ndarray[:] *= -1.0
        left_sampler.draw_cbc()

        self._show_world(world)


class Ellipse(object):
    def __init__(self, a=2.0, b=1.0):
        self.a = a
        self.b = b

    def calc_points(self, npoint):
        t_array = np.linspace(0.0, 2.0 * np.pi, npoint + 1, dtype='float64')
        point_pad = core.PointPadFp64(ndim=2, nelem=npoint + 1)
        for index, t_value in enumerate(t_array):
            x_value = self.a * np.cos(t_value)
            y_value = self.b * np.sin(t_value)
            point_pad.set_at(index, x_value, y_value)
        return point_pad


class EllipseSampler(object):
    def __init__(self, world, ellipse):
        self.world = world
        self.ellipse = ellipse
        self.points = None

    def populate_points(self, npoint=100, fac=1.0, off_x=0.0, off_y=0.0):
        self.points = _populate_sampler_points(
            self.ellipse, npoint=npoint, fac=fac, off_x=off_x, off_y=off_y)

    def draw_cbc(self, spacing=0.01):
        _draw_piecewise_bezier(self.world, self.points, spacing=spacing)


class Parabola(object):
    def __init__(self, a=0.5, t_min=-3.0, t_max=3.0):
        self.a = a
        self.t_min = t_min
        self.t_max = t_max

    def calc_points(self, npoint):
        t_array = np.linspace(self.t_min, self.t_max, npoint + 1,
                              dtype='float64')
        point_pad = core.PointPadFp64(ndim=2, nelem=npoint + 1)
        for index, t_value in enumerate(t_array):
            x_value = t_value
            y_value = self.a * t_value * t_value
            point_pad.set_at(index, x_value, y_value)
        return point_pad


class ParabolaSampler(object):
    def __init__(self, world, parabola):
        self.world = world
        self.parabola = parabola
        self.points = None

    def populate_points(self, npoint=100, fac=1.0, off_x=0.0, off_y=0.0):
        self.points = _populate_sampler_points(
            self.parabola, npoint=npoint, fac=fac, off_x=off_x, off_y=off_y)

    def draw_cbc(self, spacing=0.01):
        _draw_piecewise_bezier(self.world, self.points, spacing=spacing)


class Hyperbola(object):
    def __init__(self, a=1.0, b=1.0, t_min=-2.0, t_max=2.0):
        self.a = a
        self.b = b
        self.t_min = t_min
        self.t_max = t_max

    def calc_points(self, npoint):
        t_array = np.linspace(self.t_min, self.t_max, npoint + 1,
                              dtype='float64')
        point_pad = core.PointPadFp64(ndim=2, nelem=npoint + 1)
        for index, t_value in enumerate(t_array):
            x_value = self.a * np.cosh(t_value)
            y_value = self.b * np.sinh(t_value)
            point_pad.set_at(index, x_value, y_value)
        return point_pad


class HyperbolaSampler(object):
    def __init__(self, world, hyperbola):
        self.world = world
        self.hyperbola = hyperbola
        self.points = None

    def populate_points(self, npoint=100, fac=1.0, off_x=0.0, off_y=0.0):
        self.points = _populate_sampler_points(
            self.hyperbola, npoint=npoint, fac=fac, off_x=off_x, off_y=off_y)

    def draw_cbc(self, spacing=0.01):
        _draw_piecewise_bezier(self.world, self.points, spacing=spacing)


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
