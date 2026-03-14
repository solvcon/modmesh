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
