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


def _count_colored(image, threshold=100):
    """Count pixels brighter than the dark clear color in any channel."""
    array = _rgb_array(image)
    return int((array.max(axis=2) > threshold).sum())


def _count_reddish(image):
    """Count strongly red pixels (the first boundary set's highlight color),
    excluding the bright-in-every-channel wireframe."""
    array = _rgb_array(image)
    mask = ((array[:, :, 0] > 150) & (array[:, :, 1] < 120)
            & (array[:, :, 2] < 120))
    return int(mask.sum())


def _make_color_field():
    """A Gouraud-shaded quad (two triangles) with distinct corner colors."""
    import numpy as np
    vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        dtype='float32')
    colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
                      dtype='float32')
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='uint32')
    return vertices, colors, indices


def _update_field(widget, vertices, colors, indices):
    """Wrap numpy tables in solvcon arrays and push them to the widget."""
    core = solvcon.core
    widget.updateColorField(
        core.SimpleArrayFloat32(array=vertices.astype('float32')),
        core.SimpleArrayFloat32(array=colors.astype('float32')),
        core.SimpleArrayUint32(array=indices.astype('uint32')))


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


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetFieldTC(unittest.TestCase):
    """Field coloring and boundary highlight (step 3)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_color_field_renders(self):
        """updateColorField draws per-vertex-colored triangles."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, colors, indices = _make_color_field()
        _update_field(widget, vertices, colors, indices)
        image = _grab_or_skip(widget)
        self.assertGreater(_count_colored(image), 0)

    def test_color_field_is_swappable(self):
        """A second updateColorField replaces the first and still renders."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, colors, indices = _make_color_field()
        _update_field(widget, vertices, colors, indices)
        # Swap in a dimmer field; the latest field must render. A single grab
        # after the swap keeps the offscreen capture deterministic.
        _update_field(widget, vertices, colors * 0.5, indices)
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_show_boundary_highlights_set(self):
        """showBoundary draws the set's colored ribbon and hides it again.

        Each state grabs a freshly configured widget: a single capture of a
        fully-set-up widget is exact, matching the live screenshot path.
        """
        base_widget = pilot.RDomainWidget()
        base_widget.resize(320, 240)
        base_widget.updateMesh(_make_2d_mesh())
        base = _count_reddish(_grab_or_skip(base_widget))

        shown_widget = pilot.RDomainWidget()
        shown_widget.resize(320, 240)
        shown_widget.updateMesh(_make_2d_mesh())
        shown_widget.showBoundary(0, True)
        shown = _count_reddish(_grab_or_skip(shown_widget))
        self.assertGreater(shown, base)

        hidden_widget = pilot.RDomainWidget()
        hidden_widget.resize(320, 240)
        hidden_widget.updateMesh(_make_2d_mesh())
        hidden_widget.showBoundary(0, True)
        hidden_widget.showBoundary(0, False)
        hidden = _count_reddish(_grab_or_skip(hidden_widget))
        self.assertLess(hidden, shown)

    def test_show_boundary_without_mesh_is_noop(self):
        """showBoundary on a widget with no mesh does nothing, not crash.

        A real highlight is hundreds of red pixels; the no-op leaves none
        beyond the odd stray edge pixel the software rasterizer emits.
        """
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.showBoundary(0, True)
        image = _grab_or_skip(widget)
        self.assertLess(_count_reddish(image), 5)

    def test_color_field_rejects_out_of_range_index(self):
        """A triangle index past the vertex count is rejected, not fed to
        the GPU as an out-of-bounds fetch."""
        import numpy as np
        widget = pilot.RDomainWidget()
        vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                            dtype='float32')
        colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype='float32')
        indices = np.array([(0, 1, 9), (0, 1, 2)], dtype='uint32')
        with self.assertRaises(ValueError):
            _update_field(widget, vertices, colors, indices)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetSceneTC(unittest.TestCase):
    """Scene framing and the fit-to-scene camera (step 4)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_fit_camera_keeps_2d_mesh_in_view(self):
        """fitCameraToScene frames a 2D mesh so its wireframe stays in view."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_frames_3d_mesh_in_perspective(self):
        """A 3D mesh is framed under the perspective projection and its
        edges render."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_frames_3d_mesh_in_portrait(self):
        """A portrait viewport pulls the perspective camera back enough that
        the 3D domain is not clipped horizontally."""
        widget = pilot.RDomainWidget()
        widget.resize(240, 320)
        widget.updateMesh(_make_3d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_without_scene_is_harmless(self):
        """fitCameraToScene on an empty widget does not crash or draw."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.fitCameraToScene()
        self.assertEqual(_count_foreground(_grab_or_skip(widget)), 0)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
