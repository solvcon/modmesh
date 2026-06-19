# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon as sc

from solvcon.plot import curve


"""
Test curve helpers that populate universe World geometry.
"""


class EllipseTC(unittest.TestCase):
    def test_default(self):
        ell = curve.Ellipse()
        self.assertEqual(ell.a, 2.0)
        self.assertEqual(ell.b, 1.0)

    def test_custom(self):
        ell = curve.Ellipse(a=3.0, b=2.0)
        self.assertEqual(ell.a, 3.0)
        self.assertEqual(ell.b, 2.0)

    def test_calc_points(self):
        ell = curve.Ellipse(a=2.0, b=1.0)
        points = ell.calc_points(10)
        self.assertEqual(points.ndim, 2)
        self.assertEqual(len(points), 11)


class CurveSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = sc.WorldFp64()
        curve.CurveSampler(w, curve.Ellipse(a=2.0, b=1.0))

    def test_draw_ellipse(self):
        w = sc.WorldFp64()
        sampler = curve.CurveSampler(w, curve.Ellipse(a=2.0, b=1.0))
        sampler.populate_points(npoint=20)
        sampler.draw_cbc()
        self.assertEqual(w.nbezier, 20)

    def test_draw_parabola(self):
        w = sc.WorldFp64()
        sampler = curve.CurveSampler(w, curve.Parabola(a=0.5))
        sampler.populate_points(npoint=20)
        sampler.draw_cbc()
        self.assertEqual(w.nbezier, 20)

    def test_draw_hyperbola(self):
        w = sc.WorldFp64()
        sampler = curve.CurveSampler(w, curve.Hyperbola(a=1.0, b=1.0))
        sampler.populate_points(npoint=20)
        sampler.draw_cbc()
        self.assertEqual(w.nbezier, 20)


class ParabolaTC(unittest.TestCase):
    def test_default(self):
        par = curve.Parabola()
        self.assertEqual(par.a, 0.5)
        self.assertEqual(par.t_min, -3.0)
        self.assertEqual(par.t_max, 3.0)

    def test_custom(self):
        par = curve.Parabola(a=1.0, t_min=-2.0, t_max=2.0)
        self.assertEqual(par.a, 1.0)
        self.assertEqual(par.t_min, -2.0)
        self.assertEqual(par.t_max, 2.0)

    def test_calc_points(self):
        par = curve.Parabola(a=0.5, t_min=-3.0, t_max=3.0)
        points = par.calc_points(20)
        self.assertEqual(len(points), 21)


class HyperbolaTC(unittest.TestCase):
    def test_default(self):
        hyp = curve.Hyperbola()
        self.assertEqual(hyp.a, 1.0)
        self.assertEqual(hyp.b, 1.0)
        self.assertEqual(hyp.t_min, -2.0)
        self.assertEqual(hyp.t_max, 2.0)

    def test_custom(self):
        hyp = curve.Hyperbola(a=2.0, b=1.5, t_min=-3.0, t_max=3.0)
        self.assertEqual(hyp.a, 2.0)
        self.assertEqual(hyp.b, 1.5)
        self.assertEqual(hyp.t_min, -3.0)
        self.assertEqual(hyp.t_max, 3.0)

    def test_calc_points(self):
        hyp = curve.Hyperbola(a=1.0, b=1.0)
        points = hyp.calc_points(50)
        self.assertEqual(len(points), 51)


class BezierSampleTC(unittest.TestCase):
    def test_s_curve(self):
        bs = curve.BezierSample.s_curve()
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (1.0, 3.0))
        self.assertEqual(bs.p2, (4.0, -1.0))
        self.assertEqual(bs.p3, (5.0, 2.0))

    def test_arch(self):
        bs = curve.BezierSample.arch()
        # The arch preset is defined to start at the origin and end at
        # x=5, y=0 so that the curve spans a fixed 5-unit horizontal range
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (1.5, 4.0))
        self.assertEqual(bs.p2, (3.5, 4.0))
        self.assertEqual(bs.p3, (5.0, 0.0))

    def test_loop(self):
        bs = curve.BezierSample.loop()
        # The loop preset shares the same endpoints as arch so that both
        # presets can be compared under identical boundary conditions;
        # the difference lies in the control points that create the loop shape
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (5.0, 3.0))
        self.assertEqual(bs.p2, (0.0, 3.0))
        self.assertEqual(bs.p3, (5.0, 0.0))


class BezierSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = sc.WorldFp64()
        bs = curve.BezierSample.arch()
        curve.BezierSampler(w, bs)

    def test_draw(self):
        w = sc.WorldFp64()
        bs = curve.BezierSample.arch()
        sampler = curve.BezierSampler(w, bs)
        # nsample=10 is small enough to keep the test fast but large enough
        # to exercise the loop body in draw() more than once, catching
        # off-by-one errors in the sampling range
        sampler.draw(nsample=10)
        # draw() adds 1 Bezier curve for the arch itself
        self.assertEqual(w.nbezier, 1)
        # With default show_control_polygon=True and show_control_points=True:
        # 3 control polygon segments + 2 cross-mark segments per control
        # point * 4 points = 11 segments total
        self.assertEqual(w.nsegment, 11)

    def test_draw_no_control_polygon(self):
        w = sc.WorldFp64()
        bs = curve.BezierSample.arch()
        sampler = curve.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_polygon=False)
        self.assertEqual(w.nbezier, 1)
        # Without control polygon: only 2 cross-mark segments per control
        # point * 4 points = 8 segments (no polygon edges)
        self.assertEqual(w.nsegment, 8)

    def test_draw_no_control_points(self):
        w = sc.WorldFp64()
        bs = curve.BezierSample.arch()
        sampler = curve.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_points=False)
        self.assertEqual(w.nbezier, 1)
        # Without control point marks: only 3 polygon edge segments
        self.assertEqual(w.nsegment, 3)

    def test_draw_curve_only(self):
        w = sc.WorldFp64()
        bs = curve.BezierSample.arch()
        sampler = curve.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_polygon=False,
                     show_control_points=False)
        self.assertEqual(w.nbezier, 1)
        # No auxiliary segments at all
        self.assertEqual(w.nsegment, 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
