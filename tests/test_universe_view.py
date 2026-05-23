# Copyright (c) 2026, An-Chi Liu <phy.tiger@gmail.com>
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


import math
import unittest

import modmesh


# Tolerance for floating-point identity checks; comfortably above
# double precision and well below any test value used here.
EPS = 1e-9


class ViewTransform2dTC(unittest.TestCase):

    def test_identity_round_trip(self):
        """At identity (zoom=1, pan=0) the only thing left in the
        transform is the +Y-up flip on screen_y. Exercises both
        directions.
        """
        v = modmesh.ViewTransform2dFp64()
        sx, sy = v.screen_from_world(3.5, -2.25)
        # identity (zoom=1, pan=0) flips Y because of the +Y-up convention.
        self.assertEqual(sx, 3.5)
        self.assertEqual(sy, 2.25)
        wx, wy = v.world_from_screen(sx, sy)
        self.assertEqual(wx, 3.5)
        self.assertEqual(wy, -2.25)

    def test_pan_translates_screen(self):
        """`pan(dx, dy)` is a pure screen-space translation: at zoom=1,
        the world origin maps to (dx, dy) on screen.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(100.0, 50.0)
        sx, sy = v.screen_from_world(0.0, 0.0)
        self.assertEqual(sx, 100.0)
        self.assertEqual(sy, 50.0)

    def test_pan_is_screen_space_under_non_unit_zoom(self):
        """`pan(dx, dy)` is screen-space at every zoom level; the delta
        is not rescaled by zoom. Without this, a drag at non-unit zoom
        would move the view at the wrong rate.
        """
        v = modmesh.ViewTransform2dFp64()
        v.zoom = 4.0
        v.pan(100.0, 50.0)
        self.assertEqual(v.pan_x, 100.0)
        self.assertEqual(v.pan_y, 50.0)
        sx, sy = v.screen_from_world(0.0, 0.0)
        self.assertEqual(sx, 100.0)
        self.assertEqual(sy, 50.0)

    def test_screen_from_world_composed_formula(self):
        """Direct check of the affine formula with non-trivial pan,
        zoom, and a non-origin world point. Pins the +Y-up flip in
        screen_from_world independently of world_from_screen, so a
        symmetric sign-flip bug across both functions cannot hide
        behind a round-trip test.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(10.0, 20.0)
        v.zoom = 4.0
        sx, sy = v.screen_from_world(2.0, 3.0)
        # zoom*world_x + pan_x = 4*2 + 10 = 18
        self.assertEqual(sx, 18.0)
        # pan_y - zoom*world_y = 20 - 4*3 = 8  (+Y-up flip)
        self.assertEqual(sy, 8.0)

    def test_zoom_scales_around_origin(self):
        """`zoom_at(factor, 0, 0)` scales world distances by factor in
        screen pixels with the screen origin held fixed. The +Y flip
        still applies.
        """
        v = modmesh.ViewTransform2dFp64()
        v.zoom_at(2.0, 0.0, 0.0)
        self.assertEqual(v.zoom, 2.0)
        sx, sy = v.screen_from_world(1.0, 1.0)
        self.assertEqual(sx, 2.0)
        self.assertEqual(sy, -2.0)

    def test_zoom_keeps_cursor_world_coordinate_fixed(self):
        """The defining invariant of zoom_at: the world point that was
        under the cursor before the zoom must still be under the cursor
        after the zoom. This is what makes wheel-zoom feel right in
        the GUI.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(120.0, 80.0)
        v.zoom_at(1.5, 200.0, 175.0)
        # The world coordinate that was under the cursor before the zoom
        # should still be under the cursor after the zoom.
        prior = modmesh.ViewTransform2dFp64()
        prior.pan(120.0, 80.0)
        before_x, before_y = prior.world_from_screen(200.0, 175.0)
        after_x, after_y = v.world_from_screen(200.0, 175.0)
        self.assertAlmostEqual(before_x, after_x, delta=EPS)
        self.assertAlmostEqual(before_y, after_y, delta=EPS)

    def test_repeated_zoom_compounds(self):
        """Two zoom_at calls at the same anchor compose
        multiplicatively into a single zoom of factor1*factor2, and the
        anchor-preservation invariant survives the compounding.
        """
        v = modmesh.ViewTransform2dFp64()
        v.zoom_at(2.0, 100.0, 100.0)
        v.zoom_at(2.0, 100.0, 100.0)
        self.assertEqual(v.zoom, 4.0)
        # The cursor world coordinate (which is (100, -100) at zoom=1,
        # pan=0 because the +Y flip turns screen y=100 into world
        # y=-100) must continue to map to (100, 100) on screen.
        sx, sy = v.screen_from_world(100.0, -100.0)
        self.assertAlmostEqual(sx, 100.0, delta=EPS)
        self.assertAlmostEqual(sy, 100.0, delta=EPS)

    def test_zoom_at_rejects_non_positive(self):
        """factor <= 0 is silently rejected (no exception, zoom
        unchanged). Zero would invert the affine map; negative would
        mirror, which the widget never wants from a wheel notch.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 3.0
        v.zoom_at(0.0, 10.0, 10.0)
        self.assertEqual(v.zoom, 3.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at(-1.0, 10.0, 10.0)
        self.assertEqual(v.zoom, 3.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)

    def test_reset_returns_to_identity(self):
        """reset() returns the transform to the default-constructed
        identity state regardless of how much prior pan/zoom was
        applied.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom_at(3.0, 0.0, 0.0)
        v.reset()
        self.assertEqual(v.pan_x, 0.0)
        self.assertEqual(v.pan_y, 0.0)
        self.assertEqual(v.zoom, 1.0)

    def test_zoom_at_rejects_non_finite(self):
        """inf and NaN factors are silently rejected. A non-finite
        zoom would poison every subsequent screen<->world calculation,
        so the math layer refuses the input before it can propagate.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 2.0
        v.zoom_at(math.inf, 5.0, 5.0)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at(math.nan, 5.0, 5.0)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)

    def test_zoom_at_rejects_non_finite_anchor(self):
        """Non-finite anchor coordinates are silently rejected. Without
        this guard, world_from_screen(NaN, NaN) would yield NaN world
        coordinates and the subsequent pan recomputation would poison
        the transform; recovery would require an explicit reset().
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 2.0
        v.zoom_at(1.5, math.nan, 10.0)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at(1.5, 10.0, math.inf)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)

    def test_zoom_at_rejects_overflow(self):
        """A finite, positive factor whose product m_zoom*factor would
        overflow to infinity is silently rejected. The math layer's
        zoom < inf invariant must hold for any caller that bypasses
        setViewTransform's widget-side clamping.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 1.0e300
        v.zoom_at(1.0e10, 5.0, 5.0)  # desired = 1e310 -> overflows to inf
        self.assertEqual(v.zoom, 1.0e300)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)

    def test_zoom_at_clamped_honors_bounds(self):
        """zoom_at_clamped truncates a request that would overshoot
        the band. Here a 5x request from zoom=10 would land at 50, but
        max_zoom=20 caps it at exactly 20.
        """
        v = modmesh.ViewTransform2dFp64()
        v.zoom = 10.0
        v.zoom_at_clamped(5.0, 0.0, 0.0, 0.1, 20.0)
        self.assertEqual(v.zoom, 20.0)

    def test_zoom_at_clamped_no_op_at_max_leaves_pan_untouched(self):
        """At the max-zoom limit, an additional zoom-in request becomes
        a true no-op. The important assertion is that pan does *not*
        drift: a naive 'compute target zoom, then apply anchor shift'
        would still move pan even when zoom is clamped, which would
        feel like rubber-banding in the GUI.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 100.0
        pan_x_before = v.pan_x
        pan_y_before = v.pan_y
        # Already at max; another zoom-in must not move pan or zoom.
        v.zoom_at_clamped(2.0, 200.0, 175.0, 0.01, 100.0)
        self.assertEqual(v.zoom, 100.0)
        self.assertEqual(v.pan_x, pan_x_before)
        self.assertEqual(v.pan_y, pan_y_before)

    def test_zoom_at_clamped_no_op_at_min_leaves_pan_untouched(self):
        """Symmetric to the max-clamp test: at the min-zoom limit, a
        further zoom-out request must leave both zoom and pan
        untouched.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 0.01
        pan_x_before = v.pan_x
        pan_y_before = v.pan_y
        v.zoom_at_clamped(0.5, 200.0, 175.0, 0.01, 100.0)
        self.assertEqual(v.zoom, 0.01)
        self.assertEqual(v.pan_x, pan_x_before)
        self.assertEqual(v.pan_y, pan_y_before)

    def test_zoom_at_clamped_anchors_at_clamped_boundary(self):
        """Cursor-anchor invariant must survive a clamped step, not
        just a clean zoom_at. The wheel-zoom path goes through
        zoom_at_clamped, so any regression here would show up as a
        visible cursor drift at the band boundary in the GUI.
        """
        # Step from near-max into the clamp; the cursor world
        # coordinate must stay pinned even though the zoom request was
        # truncated.
        v = modmesh.ViewTransform2dFp64()
        v.pan(120.0, 80.0)
        v.zoom = 50.0
        anchor_x = 200.0
        anchor_y = 175.0
        before_x, before_y = v.world_from_screen(anchor_x, anchor_y)
        v.zoom_at_clamped(1000.0, anchor_x, anchor_y, 0.01, 100.0)
        self.assertEqual(v.zoom, 100.0)
        after_x, after_y = v.world_from_screen(anchor_x, anchor_y)
        self.assertAlmostEqual(before_x, after_x, delta=EPS)
        self.assertAlmostEqual(before_y, after_y, delta=EPS)

    def test_zoom_at_clamped_rejects_non_finite_or_bad_bounds(self):
        """zoom_at_clamped shares zoom_at's non-finite rejection and
        adds its own input validation on min_zoom/max_zoom. Each
        branch below covers one rejection path; in all cases zoom must
        stay unchanged.
        """
        v = modmesh.ViewTransform2dFp64()
        v.pan(50.0, 70.0)
        v.zoom = 2.0
        v.zoom_at_clamped(math.inf, 0.0, 0.0, 0.1, 10.0)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at_clamped(2.0, 0.0, 0.0, math.nan, 10.0)
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at_clamped(2.0, 0.0, 0.0, 5.0, 1.0)  # max < min
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)
        v.zoom_at_clamped(2.0, 0.0, 0.0, 0.0, 10.0)  # min not > 0
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.pan_x, 50.0)
        self.assertEqual(v.pan_y, 70.0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
