# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Curve sampling helpers without GUI dependencies.
"""

import numpy as np

from .. import core

__all__ = [
    'CurveSampler',
    'Ellipse',
    'Parabola',
    'Hyperbola',
    'BezierSample',
    'BezierSampler',
]


class CurveSampler:
    """
    Sample analytic curves into points and draw them as cubic Bezier chains.
    """

    def __init__(self, world, curve):
        self.world = world
        self.curve = curve
        self.points = None

    def populate_points(self, npoint=100, fac=1.0, off_x=0.0, off_y=0.0):
        """
        Populate sampled curve points and apply an affine transform.

        npoint controls sampling density.
        fac is a uniform scale factor.
        off_x and off_y are translation offsets in x and y.
        """
        if npoint < 1:
            raise ValueError("npoint must be at least 1")

        self.points = self.curve.calc_points(npoint)
        self.points.x.ndarray[:] = self.points.x.ndarray * fac + off_x
        self.points.y.ndarray[:] = self.points.y.ndarray * fac + off_y

    def draw_cbc(self, spacing=0.01):
        """
        Draw sampled points as a cubic Bezier chain.

        spacing is the target chord-length step used to choose per-segment
        Bezier sampling density. Smaller spacing produces denser rendering.
        """
        if self.points is None:
            raise RuntimeError(
                "populate_points() must be called before draw_cbc()"
            )
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        if len(self.points) < 2:
            return

        point_type = core.Point3dFp64
        point_x = self.points.x.ndarray
        point_y = self.points.y.ndarray
        segment_length = np.hypot(
            point_x[:-1] - point_x[1:],
            point_y[:-1] - point_y[1:],
        )
        # Minimum of 2 so very short segments are still visible.
        nsample = np.maximum((segment_length // spacing).astype(int) - 1, 2)

        for index in range(len(self.points) - 1):
            p0 = np.array(self.points[index])
            p3 = np.array(self.points[index + 1])
            delta = p3 - p0
            # Place interior cubic control points at 1/3 and 2/3 so each
            # cubic segment represents a straight line between p0 and p3.
            p1 = p0 + (1.0 / 3.0) * delta
            p2 = p0 + (2.0 / 3.0) * delta
            bezier = self.world.add_bezier(
                p0=point_type(p0[0], p0[1], 0.0),
                p1=point_type(p1[0], p1[1], 0.0),
                p2=point_type(p2[0], p2[1], 0.0),
                p3=point_type(p3[0], p3[1], 0.0),
            )
            bezier.sample(int(nsample[index]))


class Ellipse:
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


class Parabola:
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


class Hyperbola:
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
