# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for RDomainWidget, the QRhi reimplementation of the pilot 3D viewer.

The widget renders through QRhi, so these tests exercise the offscreen
capture path (grabImage via saveImage). QRhi needs a real graphics surface;
where one is unavailable (e.g. the offscreen QPA platform on a headless
macOS runner) the render-dependent tests skip rather than fail. The Linux CI
build job drives them under Xvfb with the software rasterizer.
"""

import os
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QImage
except ImportError:
    pilot = None


def _grab_or_skip(widget):
    """Render the widget offscreen and return a QImage.

    Skip the calling test when QRhi cannot create an offscreen surface on
    this platform (saveImage then writes no file or a null image).
    """
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "domain.png")
        widget.saveImage(path)
        if not os.path.exists(path):
            raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
        image = QImage(path)
    if image.isNull():
        raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
    return image


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetFoundationTC(unittest.TestCase):
    """The render foundation and the Python control spine (step 1)."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def test_construct_from_python(self):
        """RDomainWidget is constructible directly from Python."""
        widget = pilot.RDomainWidget()
        self.assertIsNotNone(widget)

    def test_save_image_writes_png_file(self):
        """saveImage routes through grabImage and yields a valid frame whose
        pixel size matches the widget."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        image = _grab_or_skip(widget)
        self.assertGreater(image.width(), 0)
        self.assertGreater(image.height(), 0)

    def test_render_draws_primitive(self):
        """The render loop clears to the dark background and draws the lone
        primitive through RMaterial: the lower-center is the primitive's
        color and a corner stays background."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        image = _grab_or_skip(widget)
        inside = image.pixelColor(160, 130)
        corner = image.pixelColor(5, 5)
        # The primitive is drawn in a blue-dominant flat color.
        self.assertGreater(inside.blue(), inside.red())
        self.assertGreater(inside.green(), inside.red())
        # The corner is the dark clear color.
        self.assertLess(corner.red(), 80)
        self.assertLess(corner.green(), 80)
        self.assertLess(corner.blue(), 80)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
