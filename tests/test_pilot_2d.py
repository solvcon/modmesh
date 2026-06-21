# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for R2DWidget and its on-screen screenshot APIs.
"""

import os
import tempfile
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QGuiApplication, QImage
except ImportError:
    pilot = None

_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

# The pixel helpers below classify colors that RWorldRenderer2d paints (see
# RWorldRenderer2d.cpp). GEOMETRY (120, 180, 240) is the only strongly-blue
# color: the backdrop, grid, axes, and origin marker all have blue <= 80, so
# a blue-dominant pixel must be geometry. ORIGIN (220, 80, 80) is the only
# red-dominant color, since the yellow axes have equal red and green, which
# lets the origin marker be located on its own.


def _png_size(data):
    """Return PNG width and height from IHDR (offsets 16 and 20)."""
    assert data[:8] == _PNG_MAGIC
    return (int.from_bytes(data[16:20], 'big'),
            int.from_bytes(data[20:24], 'big'))


def _load_rgba(path):
    """Load a PNG into an (height, width, 4) RGBA uint8 array of pixels.

    The pixels are physical (device) pixels: a HiDPI capture is larger than
    the widget's logical size by the device-pixel ratio.
    """
    img = QImage(path)
    assert not img.isNull(), "QImage failed to load %s" % path
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = img.width(), img.height()
    # bytesPerLine may pad the scanline past width * 4, so reshape on the
    # full stride (in pixels) and then slice the padding off.
    stride = img.bytesPerLine() // 4
    arr = np.frombuffer(bytes(img.constBits()), dtype='uint8')
    return arr.reshape(height, stride, 4)[:, :width, :].copy()


def _geometry_mask(arr):
    """Return the boolean mask of GEOMETRY-colored (blue-dominant) pixels."""
    red = arr[:, :, 0].astype('int32')
    green = arr[:, :, 1].astype('int32')
    blue = arr[:, :, 2].astype('int32')
    return (blue >= 120) & (blue > red + 30) & (blue > green)


def _origin_mask(arr):
    """Return the boolean mask of ORIGIN-marker (red-dominant) pixels."""
    red = arr[:, :, 0].astype('int32')
    green = arr[:, :, 1].astype('int32')
    blue = arr[:, :, 2].astype('int32')
    return (red >= 150) & (red > green + 40) & (red > blue + 40)


def _has_geometry_near(mask, px, py, radius=4):
    """Return whether a geometry pixel lies within radius of (px, py).

    A point that rounds to outside the image has no geometry by definition,
    so return False rather than sampling the clamped edge window.
    """
    col, row = int(round(px)), int(round(py))
    height, width = mask.shape
    if not (0 <= col < width and 0 <= row < height):
        return False
    window = mask[max(0, row - radius):row + radius + 1,
                  max(0, col - radius):col + radius + 1]
    return bool(window.any())


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

    def _render_world(self, world, pan_x, pan_y, zoom):
        """Render world under an explicit view and return (view, mask, dpr).

        Set pan/zoom on the widget (an explicit transform also disables the
        widget's auto-centering, fixing the world->screen mapping regardless
        of the widget's size), capture the widget to a PNG, and return the
        view, the GEOMETRY-pixel mask, and the device-pixel ratio recovered
        from the origin marker. The dpr is recovered from the rendered marker
        rather than assumed, so physical-pixel predictions hold on both
        standard and HiDPI displays.
        """
        self.widget.updateWorld(world)
        view = solvcon.ViewTransform2dFp64()
        view.pan_x = pan_x
        view.pan_y = pan_y
        view.zoom = zoom
        self.widget.setViewTransform(view)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "render.png")
            self.widget.saveImage(path)
            arr = _load_rgba(path)

        rows, cols = np.where(_origin_mask(arr))
        self.assertGreater(len(cols), 0, "origin marker was not rendered")
        dpr = (cols.mean() / pan_x + rows.mean() / pan_y) / 2.0
        return view, _geometry_mask(arr), dpr

    def test_known_world_renders_to_expected_pixels(self):
        """Gate test: a known world renders to the expected pixels.

        A single horizontal segment is rendered under an explicit view
        transform. The render must be non-blank, the segment's two endpoints
        must land where the view transform predicts, and a removed (DEAD)
        rectangle must leave no pixels in its region.
        """
        point_a = (-1.5, 1.5)
        point_b = (1.5, 1.5)
        dead_rect = (1.2, -1.2, 2.0, -0.4)  # Well clear of the live segment.

        world = solvcon.WorldFp64()
        world.add_segment(
            solvcon.Point3dFp64(point_a[0], point_a[1], 0.0),
            solvcon.Point3dFp64(point_b[0], point_b[1], 0.0))
        dead = world.add_rectangle(*dead_rect)
        world.remove_shape(dead)

        view, geometry, dpr = self._render_world(world, 110.0, 110.0, 24.0)

        # Non-blank: the live segment painted real geometry pixels.
        self.assertGreater(int(geometry.sum()), 0)

        # Known endpoints map to the expected pixels under the view transform.
        for label, point in (("A", point_a), ("B", point_b)):
            screen_x, screen_y = view.screen_from_world(point[0], point[1])
            self.assertTrue(
                _has_geometry_near(geometry, screen_x * dpr, screen_y * dpr),
                "no geometry at predicted endpoint %s" % label)

        # The removed rectangle's four edges must be culled: no geometry
        # anywhere in its bounding box.
        corners = ((dead_rect[0], dead_rect[1]), (dead_rect[2], dead_rect[1]),
                   (dead_rect[2], dead_rect[3]), (dead_rect[0], dead_rect[3]))
        screen_x, screen_y = zip(*(view.screen_from_world(cx, cy)
                                   for cx, cy in corners))
        px = [sx * dpr for sx in screen_x]
        py = [sy * dpr for sy in screen_y]
        region = geometry[max(0, int(min(py)) - 2):int(max(py)) + 3,
                          max(0, int(min(px)) - 2):int(max(px)) + 3]
        self.assertEqual(int(region.sum()), 0, "removed shape was painted")

    def test_circle_renders_as_hollow_loop_on_locus(self):
        """A circle renders as a closed, hollow ring on its locus.

        add_circle builds the outline from four cubic Beziers, stroked and
        not filled. Sampling the predicted ring at twelve angles checks the
        loop is painted all the way around -- a dropped Bezier quadrant or an
        open arc would leave a gap -- and that includes the four cardinal
        points landing where the view transform predicts. The center and a
        mid-radius point stay blank because the renderer strokes outlines
        only.
        """
        center = (0.5, 0.3)
        radius = 2.0

        world = solvcon.WorldFp64()
        world.add_circle(center[0], center[1], radius)

        view, geometry, dpr = self._render_world(world, 160.0, 120.0, 30.0)

        # The ring is painted all the way around, cardinal points included.
        for degrees in range(0, 360, 30):
            angle = np.radians(degrees)
            world_x = center[0] + radius * np.cos(angle)
            world_y = center[1] + radius * np.sin(angle)
            screen_x, screen_y = view.screen_from_world(world_x, world_y)
            self.assertTrue(
                _has_geometry_near(geometry, screen_x * dpr, screen_y * dpr),
                "no geometry on the ring at %d degrees" % degrees)

        # Outlines only: the center and a mid-radius point stay blank.
        for world_x, world_y in (center, (center[0] + radius / 2, center[1])):
            screen_x, screen_y = view.screen_from_world(world_x, world_y)
            self.assertFalse(
                _has_geometry_near(geometry, screen_x * dpr, screen_y * dpr),
                "circle interior should be hollow")


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
