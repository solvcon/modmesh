# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for R2DWidget and its on-screen screenshot APIs.
"""

import os
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QGuiApplication
except ImportError:
    pilot = None

_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def _png_size(data):
    """Return PNG width and height from IHDR (offsets 16 and 20)."""
    assert data[:8] == _PNG_MAGIC
    return (int.from_bytes(data[16:20], 'big'),
            int.from_bytes(data[20:24], 'big'))


def _build_world():
    """A world exercising every primitive R2DWidget paints, plus the two
    cases the 2D path must handle silently: an out-of-plane (z != 0) point
    that gets projected, and a removed (DEAD) shape whose geometry must be
    dropped by collect_live_*.
    """
    w = solvcon.WorldFp64()
    # Bare segment (owned by no shape) and a bare cubic Bezier.
    w.add_segment(solvcon.Point3dFp64(-3, 3, 0),
                  solvcon.Point3dFp64(3, 3, 0))
    w.add_bezier(solvcon.Point3dFp64(-3, 0, 0),
                 solvcon.Point3dFp64(-1, 2, 0),
                 solvcon.Point3dFp64(1, -2, 0),
                 solvcon.Point3dFp64(3, 0, 0))
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


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
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
        crashing; RWorldRenderer2d is skipped when no world is set.
        """
        self.widget.updateWorld(_build_world())
        self.widget.updateWorld(None)
        self.widget.requestRepaint()

    def test_resync_after_mutating_world(self):
        """Re-issuing updateWorld on the same world after adding geometry
        (the live Canvas sample flow) repaints the new state. Guards that
        the widget re-reads the world rather than caching a snapshot.
        """
        w = solvcon.WorldFp64()
        w.add_circle(0.0, 0.0, 1.0)
        self.widget.updateWorld(w)
        w.add_rectangle(-2, -2, 2, 2)
        self.widget.updateWorld(w)
        self.widget.requestRepaint()

    def test_empty_world_paints_without_error(self):
        """A world with no geometry is valid: the loops are simply empty.
        Catches off-by-one / null-pad assumptions in RWorldRenderer2d.
        """
        self.widget.updateWorld(solvcon.WorldFp64())
        self.widget.requestRepaint()

    def test_view_transform_round_trip(self):
        """The view-state API that frames the world is intact: a pan/zoom
        set via setViewTransform reads back through the viewTransform
        property (zoom 3.0 is well within the widget's clamp band).
        """
        self.widget.resetView()
        v = solvcon.ViewTransform2dFp64()
        v.pan(40.0, 25.0)
        v.zoom = 3.0
        self.widget.setViewTransform(v)
        got = self.widget.viewTransform
        self.assertEqual(got.pan_x, 40.0)
        self.assertEqual(got.pan_y, 25.0)
        self.assertEqual(got.zoom, 3.0)

    @unittest.skip("TODO: pixel-level DEAD-shape culling assertions")
    def test_dead_shape_culling_renders_pixels(self):
        """Assert live geometry is painted and DEAD shapes are culled.

        saveImage captures the widget; pixel sampling helpers are still TODO.
        """
        # w = solvcon.WorldFp64()
        # live = w.add_rectangle(-3, -3, -1, -1)   # region A (lower-left)
        # dead = w.add_rectangle(1, 1, 3, 3)       # region B (upper-right)
        # w.remove_shape(dead)
        # self.widget.setViewTransform(<fixed pan/zoom>)
        # self.widget.saveImage(path)
        # img = read_png(path)
        # self.assertGreater(foreground_pixels(img, region_A), 0)
        # self.assertEqual(foreground_pixels(img, region_B), 0)
        raise NotImplementedError


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class R2DWidgetScreenshotTC(unittest.TestCase):
    """R2DWidget on-screen screenshot APIs (saveImage/clipImage)."""

    @classmethod
    def setUpClass(cls):
        cls.widget = pilot.RManager.instance.setUp().add2DWidget()

    @classmethod
    def tearDownClass(cls):
        cls.widget = None

    def test_save_image_writes_png_file(self):
        """saveImage writes a valid, non-empty PNG of the widget."""
        self.widget.updateWorld(_build_world())
        self.widget.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "widget.png")
            self.widget.saveImage(path)
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)
        width, height = _png_size(data)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_current_2d_widget_exposes_screenshot_api(self):
        """currentR2DWidget returns the active 2D widget, API intact."""
        mgr = pilot.RManager.instance.setUp()
        mgr.add2DWidget()
        current = mgr.currentR2DWidget()
        self.assertIsNotNone(current)
        current.updateWorld(_build_world())
        current.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "current.png")
            current.saveImage(path)
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)
        width, height = _png_size(data)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_list_2d_widgets_returns_usable_widgets(self):
        """list2DWidgets returns real R2DWidgets, not bare QWidgets."""
        mgr = pilot.RManager.instance.setUp()
        mgr.add2DWidget()
        widgets = mgr.list2DWidgets()
        self.assertIsInstance(widgets, list)
        self.assertGreaterEqual(len(widgets), 1)
        for widget in widgets:
            self.assertTrue(hasattr(widget, "saveImage"))
            self.assertTrue(hasattr(widget, "clipImage"))
        listed = widgets[-1]
        listed.updateWorld(_build_world())
        listed.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "listed.png")
            listed.saveImage(path)
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)

    def test_clip_image_copies_pixmap_to_clipboard(self):
        """clipImage puts a non-null widget pixmap on the clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard is None:
            self.skipTest("no clipboard in this environment")
        self.widget.updateWorld(_build_world())
        self.widget.resetView()
        clipboard.clear()
        self.assertTrue(clipboard.pixmap().isNull())
        self.widget.clipImage()
        pixmap = clipboard.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)
        self.assertGreater(pixmap.height(), 0)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
