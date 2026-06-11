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

"""
Tests for the issue #754 PR2 2D canvas: R2DWidget rendering a World<double>.

The widget paints with QPainter from the world's backend-independent
collect_live_* surface, so these run inside the pilot binary (Qt available)
via ``make run_pilot_pytest PYTEST_OPTS=tests/test_pilot_2d.py``. They guard
the C++ -> Python driving path (updateWorld accepting realistic geometry
without raising) and the view-state API; pixel-level appearance is checked
manually in the GUI. See test_dead_shape_culling_renders_pixels (skipped) for
the planned pixel-level coverage and what it needs first.
"""

import unittest

import modmesh

try:
    from modmesh import pilot
except ImportError:
    pilot = None


def _build_world():
    """A world exercising every primitive R2DWidget paints, plus the two
    cases the 2D path must handle silently: an out-of-plane (z != 0) point
    that gets projected, and a removed (DEAD) shape whose geometry must be
    dropped by collect_live_*.
    """
    w = modmesh.WorldFp64()
    # Bare segment (owned by no shape) and a bare cubic Bezier.
    w.add_segment(modmesh.Point3dFp64(-3, 3, 0),
                  modmesh.Point3dFp64(3, 3, 0))
    w.add_bezier(modmesh.Point3dFp64(-3, 0, 0),
                 modmesh.Point3dFp64(-1, 2, 0),
                 modmesh.Point3dFp64(1, -2, 0),
                 modmesh.Point3dFp64(3, 0, 0))
    # Shapes: segment-backed and Bezier-backed.
    w.add_triangle(0, 0, 1, 0, 0, 1)
    w.add_rectangle(2, 2, 4, 3)
    w.add_circle(-2, -2, 1.0)
    # A point off the z=0 plane: the 2D widget drops z, must not error.
    w.add_point(0.5, 0.5, 5.0)
    # A removed shape: its segments must be culled, not painted.
    dead = w.add_triangle(8, 8, 9, 8, 8, 9)
    w.remove_shape(dead)
    return w


@unittest.skipUnless(modmesh.HAS_PILOT, "Qt pilot is not built")
class R2DWidgetWorldTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.widget = pilot.RManager.instance.setUp().add2DWidget()

    @classmethod
    def tearDownClass(cls):
        cls.widget = None

    def test_add_2d_widget_created(self):
        """The 2D widget is created and accessible via the manager.
        """
        self.assertIsNotNone(self.widget)

    def test_update_world_accepts_mixed_geometry(self):
        """updateWorld accepts a world with mixed geometry (segments, curves,
        points, shapes) and a removed shape, without raising.
        """
        self.widget.updateWorld(_build_world())
        self.widget.requestRepaint()

    def test_update_world_none_clears(self):
        """A null world clears the canvas to its grid backdrop instead of
        crashing; paintWorld early-returns when no world is set.
        """
        self.widget.updateWorld(_build_world())
        self.widget.updateWorld(None)
        self.widget.requestRepaint()

    def test_resync_after_mutating_world(self):
        """Re-issuing updateWorld on the same world after adding geometry
        (the live Canvas sample flow) repaints the new state. Guards that
        the widget re-reads the world rather than caching a snapshot.
        """
        w = modmesh.WorldFp64()
        w.add_circle(0.0, 0.0, 1.0)
        self.widget.updateWorld(w)
        w.add_rectangle(-2, -2, 2, 2)
        self.widget.updateWorld(w)
        self.widget.requestRepaint()

    def test_empty_world_paints_without_error(self):
        """A world with no geometry is valid: the loops are simply empty.
        Catches off-by-one / null-pad assumptions in paintWorld.
        """
        self.widget.updateWorld(modmesh.WorldFp64())
        self.widget.requestRepaint()

    def test_view_transform_round_trip(self):
        """The view-state API that frames the world is intact: a pan/zoom
        set via setViewTransform reads back through the viewTransform
        property (zoom 3.0 is well within the widget's clamp band).
        """
        self.widget.resetView()
        v = modmesh.ViewTransform2dFp64()
        v.pan(40.0, 25.0)
        v.zoom = 3.0
        self.widget.setViewTransform(v)
        got = self.widget.viewTransform
        self.assertEqual(got.pan_x, 40.0)
        self.assertEqual(got.pan_y, 25.0)
        self.assertEqual(got.zoom, 3.0)

    @unittest.skip("TODO: needs a synchronous-render hook on R2DWidget")
    def test_dead_shape_culling_renders_pixels(self):
        """Assert live geometry is painted and DEAD shapes are culled.

        Needs a synchronous-render hook (saveImage, parity with
        R3DWidget) to grab the canvas headless. Skipped until then.
        """
        # w = modmesh.WorldFp64()
        # live = w.add_rectangle(-3, -3, -1, -1)   # region A (lower-left)
        # dead = w.add_rectangle(1, 1, 3, 3)       # region B (upper-right)
        # w.remove_shape(dead)
        # self.widget.setViewTransform(<fixed pan/zoom>)
        # img = self.widget.saveImage(width=400, height=400)  # future hook
        # self.assertGreater(foreground_pixels(img, region_A), 0)  # painted
        # self.assertEqual(foreground_pixels(img, region_B), 0)    # culled
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
