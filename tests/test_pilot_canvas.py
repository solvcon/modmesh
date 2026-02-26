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


import unittest

import modmesh as mm
import pytest

pytest.importorskip("PySide6")

from modmesh.pilot import _canvas  # noqa: E402


class EllipseTC(unittest.TestCase):
    def test_npoint(self):
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        points = ell.calc_points(10)
        # ndim must be 2 because the ellipse lives in the xy-plane
        self.assertEqual(points.ndim, 2)
        # calc_points(n) samples n+1 points so that n intervals span [0, 2pi]
        self.assertEqual(len(points), 11)

    def test_closure(self):
        """First and last point should coincide for a full ellipse."""
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        # 100 intervals gives fine angular resolution (~3.6 deg per step)
        # while keeping the test fast
        points = ell.calc_points(100)
        p_first = points.get_at(0)
        p_last = points.get_at(len(points) - 1)
        # t=0 and t=2*pi map to the same point analytically, so the
        # round-trip through float64 trig should agree to near machine
        # precision; places=10 allows for ~1e-10 floating-point error
        self.assertAlmostEqual(p_first.x, p_last.x, places=10)
        self.assertAlmostEqual(p_first.y, p_last.y, places=10)

    def test_axes(self):
        """Check that extreme points match semi-axes."""
        ell = _canvas.Ellipse(a=3.0, b=2.0)
        # 400 intervals gives an angular step of ~0.9 deg, so the sampled
        # maximum deviates from the true semi-axis by at most
        # a*(1 - cos(pi/400)) < 3e-4, which is within places=2 (1e-2)
        points = ell.calc_points(400)
        xs = points.x.ndarray
        ys = points.y.ndarray
        # x(t) = a*cos(t) has maximum a at t=0; y(t) = b*sin(t) has
        # maximum b at t=pi/2
        self.assertAlmostEqual(float(max(xs)), 3.0, places=2)
        self.assertAlmostEqual(float(max(ys)), 2.0, places=2)


class EllipseSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        _canvas.EllipseSampler(w, ell)

    def test_populate_and_draw(self):
        w = mm.WorldFp64()
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        sampler = _canvas.EllipseSampler(w, ell)
        # Use non-trivial fac/off_x/off_y values to exercise the scaling
        # and translation paths in populate_points; npoint=20 keeps the
        # test fast while still producing multiple line segments
        sampler.populate_points(npoint=20, fac=2.0, off_x=1.0, off_y=1.0)
        sampler.draw_cbc()
        # 20 intervals produce 20 piecewise Bezier segments
        self.assertEqual(w.nbezier, 20)

    def test_draw_without_populate_raises(self):
        w = mm.WorldFp64()
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        sampler = _canvas.EllipseSampler(w, ell)
        with self.assertRaisesRegex(RuntimeError, "populate_points"):
            sampler.draw_cbc()

    def test_non_positive_spacing_raises(self):
        w = mm.WorldFp64()
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        sampler = _canvas.EllipseSampler(w, ell)
        sampler.populate_points(npoint=10)
        with self.assertRaisesRegex(ValueError, "spacing"):
            sampler.draw_cbc(spacing=0.0)

    def test_npoint_zero_raises(self):
        w = mm.WorldFp64()
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        sampler = _canvas.EllipseSampler(w, ell)
        with self.assertRaisesRegex(ValueError, "npoint"):
            sampler.populate_points(npoint=0)

    def test_scaling_and_offset(self):
        """Verify fac/off_x/off_y transform points correctly."""
        ell = _canvas.Ellipse(a=2.0, b=1.0)
        base_points = ell.calc_points(10)
        sampler = _canvas.EllipseSampler(mm.WorldFp64(), ell)
        sampler.populate_points(npoint=10, fac=2.0, off_x=5.0, off_y=3.0)
        for i in range(len(sampler.points)):
            pb = base_points.get_at(i)
            ps = sampler.points.get_at(i)
            # populate_points scales by fac then shifts by off_x/off_y
            self.assertAlmostEqual(ps.x, pb.x * 2.0 + 5.0, places=10)
            self.assertAlmostEqual(ps.y, pb.y * 2.0 + 3.0, places=10)


class ParabolaTC(unittest.TestCase):
    def test_npoint(self):
        par = _canvas.Parabola(a=0.5, t_min=-3.0, t_max=3.0)
        # Same n+1 convention as Ellipse: 20 intervals produce 21 points
        points = par.calc_points(20)
        self.assertEqual(len(points), 21)

    def test_vertex(self):
        """Vertex at t=0 should be at (0,0)."""
        par = _canvas.Parabola(a=1.0, t_min=-2.0, t_max=2.0)
        # 100 intervals over a symmetric range [-2, 2] produce 101 points;
        # np.linspace(-2, 2, 101)[50] == 0.0 exactly, so the middle point
        # always falls precisely on t=0 and thus on the vertex (0, 0)
        points = par.calc_points(100)
        mid = len(points) // 2
        p = points.get_at(mid)
        self.assertAlmostEqual(p.x, 0.0, places=5)
        self.assertAlmostEqual(p.y, 0.0, places=5)


class ParabolaSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        par = _canvas.Parabola(a=0.5)
        _canvas.ParabolaSampler(w, par)

    def test_populate_and_draw(self):
        w = mm.WorldFp64()
        par = _canvas.Parabola(a=0.5)
        sampler = _canvas.ParabolaSampler(w, par)
        sampler.populate_points(npoint=20, fac=2.0, off_x=1.0, off_y=1.0)
        sampler.draw_cbc()
        # 20 intervals produce 20 piecewise Bezier segments
        self.assertEqual(w.nbezier, 20)


class HyperbolaTC(unittest.TestCase):
    def test_npoint(self):
        hyp = _canvas.Hyperbola(a=1.0, b=1.0)
        # Same n+1 convention: 50 intervals produce 51 points
        points = hyp.calc_points(50)
        self.assertEqual(len(points), 51)

    def test_right_branch_x_positive(self):
        """All x values should be >= a (cosh >= 1)."""
        hyp = _canvas.Hyperbola(a=1.0, b=1.0)
        points = hyp.calc_points(100)
        xs = points.x.ndarray
        for x in xs:
            # x(t) = a*cosh(t); cosh(t) >= 1 for all real t, so x >= a = 1.0.
            # The 1e-10 tolerance accommodates floating-point rounding in
            # cosh() near t=0 where cosh(0) == 1.0 exactly in IEEE 754
            self.assertGreaterEqual(float(x), 1.0 - 1e-10)


class HyperbolaSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        hyp = _canvas.Hyperbola(a=1.0, b=1.0)
        _canvas.HyperbolaSampler(w, hyp)

    def test_populate_and_draw(self):
        w = mm.WorldFp64()
        hyp = _canvas.Hyperbola(a=1.0, b=1.0)
        sampler = _canvas.HyperbolaSampler(w, hyp)
        sampler.populate_points(npoint=20, fac=2.0, off_x=1.0, off_y=1.0)
        sampler.draw_cbc()
        # 20 intervals produce 20 piecewise Bezier segments
        self.assertEqual(w.nbezier, 20)


class BezierSampleTC(unittest.TestCase):
    def test_s_curve(self):
        bs = _canvas.BezierSample.s_curve()
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (1.0, 3.0))
        self.assertEqual(bs.p2, (4.0, -1.0))
        self.assertEqual(bs.p3, (5.0, 2.0))

    def test_arch(self):
        bs = _canvas.BezierSample.arch()
        # The arch preset is defined to start at the origin and end at
        # x=5, y=0 so that the curve spans a fixed 5-unit horizontal range
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (1.5, 4.0))
        self.assertEqual(bs.p2, (3.5, 4.0))
        self.assertEqual(bs.p3, (5.0, 0.0))

    def test_loop(self):
        bs = _canvas.BezierSample.loop()
        # The loop preset shares the same endpoints as arch so that both
        # presets can be compared under identical boundary conditions;
        # the difference lies in the control points that create the loop shape
        self.assertEqual(bs.p0, (0.0, 0.0))
        self.assertEqual(bs.p1, (5.0, 3.0))
        self.assertEqual(bs.p2, (0.0, 3.0))
        self.assertEqual(bs.p3, (5.0, 0.0))


class BezierSamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        bs = _canvas.BezierSample.arch()
        _canvas.BezierSampler(w, bs)

    def test_draw(self):
        w = mm.WorldFp64()
        bs = _canvas.BezierSample.arch()
        sampler = _canvas.BezierSampler(w, bs)
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
        w = mm.WorldFp64()
        bs = _canvas.BezierSample.arch()
        sampler = _canvas.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_polygon=False)
        self.assertEqual(w.nbezier, 1)
        # Without control polygon: only 2 cross-mark segments per control
        # point * 4 points = 8 segments (no polygon edges)
        self.assertEqual(w.nsegment, 8)

    def test_draw_no_control_points(self):
        w = mm.WorldFp64()
        bs = _canvas.BezierSample.arch()
        sampler = _canvas.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_points=False)
        self.assertEqual(w.nbezier, 1)
        # Without control point marks: only 3 polygon edge segments
        self.assertEqual(w.nsegment, 3)

    def test_draw_curve_only(self):
        w = mm.WorldFp64()
        bs = _canvas.BezierSample.arch()
        sampler = _canvas.BezierSampler(w, bs)
        sampler.draw(nsample=10, show_control_polygon=False,
                     show_control_points=False)
        self.assertEqual(w.nbezier, 1)
        # No auxiliary segments at all
        self.assertEqual(w.nsegment, 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
