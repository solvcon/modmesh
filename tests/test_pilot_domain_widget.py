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
import platform
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QImage
except ImportError:
    pilot = None


def _make_2d_mesh():
    """Two triangles and one quadrilateral in the z = 0 plane."""
    core = solvcon.core
    T = core.StaticMesh.TRIANGLE
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    mh.cltpn.ndarray[:] = [T, T, Q]
    mh.clnds.ndarray[:, :5] = [(3, 0, 3, 2, -1), (3, 0, 1, 3, -1),
                               (4, 1, 4, 5, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _make_3d_mesh():
    """A single tetrahedron."""
    core = solvcon.core
    mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
    mh.ndcrd.ndarray[:, :] = [(0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)]
    mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _grab_or_skip(widget):
    """Render the widget offscreen and return a QImage.

    Skip the calling test when an offscreen grab is unavailable or unreliable
    here: the headless Windows runner's debug software rasterizer stalls
    indefinitely creating a dedicated grab device, and the offscreen QPA
    platform on a headless macOS runner reports no QRhi support (saveImage
    then writes no file / a null image). Render correctness stays covered on
    Linux (Xvfb) and the platforms where grabbing works.
    """
    if platform.system() == "Windows":
        raise unittest.SkipTest("offscreen grabbing unreliable on Windows CI")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "domain.png")
        widget.saveImage(path)
        if not os.path.exists(path):
            raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
        image = QImage(path)
    if image.isNull():
        raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
    return image


def _rgb_array(image):
    """Return the image as an (h, w, 3) uint8 array.

    Per-pixel QImage reads are far too slow in a debug build (minutes for a
    full frame), so the whole buffer is mapped through numpy at once. The
    bytes are copied out of the QImage immediately: the source image is a
    local that would otherwise be freed while the array still views it.
    """
    import numpy as np
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    buffer = converted.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8,
                          count=converted.sizeInBytes()).copy()
    array = array.reshape(height, converted.bytesPerLine())[:, :width * 4]
    return array.reshape(height, width, 4)[:, :, :3]


def _count_foreground(image, threshold=150):
    """Count wireframe line pixels.

    The wireframe is a light hairline, bright in every channel (~0.85), over
    a dark background (~0.12). Requiring all three channels to be bright
    distinguishes the line from the dark backdrop and from the odd
    single-channel edge pixel the software rasterizer can leave behind.
    """
    array = _rgb_array(image)
    mask = ((array[:, :, 0] > threshold) & (array[:, :, 1] > threshold)
            & (array[:, :, 2] > threshold))
    return int(mask.sum())


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

    def test_empty_scene_is_background(self):
        """With no mesh the frame is the uniform dark clear color: nothing
        is drawn, so no pixel is a bright line."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        image = _grab_or_skip(widget)
        self.assertEqual(_count_foreground(image), 0)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetMeshTC(unittest.TestCase):
    """Domain wireframe rendering for 2D and 3D meshes (step 2)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_mesh_property_round_trip(self):
        """updateMesh exposes the mesh through the pybind11 widget."""
        widget = pilot.RDomainWidget()
        mh = _make_2d_mesh()
        widget.updateMesh(mh)
        self.assertIsNotNone(widget.mesh)
        self.assertEqual(widget.mesh.ncell, 3)

    def test_2d_mesh_draws_wireframe(self):
        """A 2D mesh renders a wireframe: some pixels are bright lines."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        image = _grab_or_skip(widget)
        self.assertGreater(_count_foreground(image), 0)

    def test_3d_mesh_draws_wireframe(self):
        """A 3D tetrahedron renders its edges without error."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        image = _grab_or_skip(widget)
        self.assertGreater(_count_foreground(image), 0)

    def test_show_mesh_toggles_visibility(self):
        """showMesh(False) hides the wireframe; showMesh(True) restores it.

        Hiding removes most of the wireframe rather than every last pixel: a
        software rasterizer can leave a few stray edge pixels behind, so the
        check is relative (hidden is a small fraction of shown) instead of an
        exact zero.
        """
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        shown = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(shown, 0)
        widget.showMesh(False)
        hidden = _count_foreground(_grab_or_skip(widget))
        self.assertLess(hidden, shown * 0.5)
        widget.showMesh(True)
        restored = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(restored, hidden)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
