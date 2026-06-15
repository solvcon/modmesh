# Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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

import modmesh

try:
    from modmesh import pilot
    from modmesh.pilot import _mesh_info
    from PySide6.QtWidgets import QApplication, QMenu
except ImportError:
    pilot = None


def _make_sample_mesh():
    """
    Two triangles and one quadrilateral; ``build_ghost`` adds ghost cells
    and nodes whose presence the panel must not count.
    """
    core = modmesh.core
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


def _make_single_triangle():
    """Return a one-cell mesh.

    Its ``ncell`` of 1 differs from the sample mesh's 3, so a test cannot
    pass on mesh state left in the shared ``RManager`` singleton.
    """
    core = modmesh.core
    mh = core.StaticMesh(ndim=2, nnode=3, nface=0, ncell=1)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1)]
    mh.cltpn.ndarray[:] = [core.StaticMesh.TRIANGLE]
    mh.clnds.ndarray[:, :4] = [(3, 0, 1, 2)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _section_map(sections):
    """Map each section name to its ``{property: value}`` dict.

    Counts and Ghost share property names (node, face, cell), so the rows
    cannot be flattened into one namespace.
    """
    return {name: dict(rows) for name, rows in sections}


def _tree_sections(tree):
    """Map each group under the mesh tree root to ``{property: value}``."""
    result = {}
    root = tree.topLevelItem(0)
    for i in range(root.childCount()):
        group = root.child(i)
        pairs = {}
        for j in range(group.childCount()):
            prop, value = group.child(j).text(0).split(": ", 1)
            pairs[prop] = value
        result[group.text(0)] = pairs
    return result


@unittest.skipUnless(modmesh.HAS_PILOT, "Qt pilot is not built")
class MakeMeshInfoTC(unittest.TestCase):
    def test_excludes_ghost_entities(self):
        info = _section_map(
            _mesh_info.MeshInfoPanel.make_mesh_info(_make_sample_mesh()))
        self.assertEqual(info["Counts"]["dim"], "2")
        self.assertEqual(info["Counts"]["node"], "6")
        self.assertEqual(info["Counts"]["cell"], "3")
        # The ghost cells must not inflate the cell-type counts.
        self.assertEqual(info["Cell types"]["triangle"], "2")
        self.assertEqual(info["Cell types"]["quadrilateral"], "1")
        # The bounding box must come from the body nodes only.
        self.assertEqual(info["Bounding box"]["x"], "[0, 2]")
        self.assertEqual(info["Bounding box"]["y"], "[0, 1]")


@unittest.skipUnless(modmesh.HAS_PILOT, "Qt pilot is not built")
class MeshInfoTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        # The "Panels" group is owned by the caller, not by MeshInfo.
        self.menu = QMenu("Panels", self.mgr.mainWindow)

    def test_current_r3dwidget_exposes_mesh(self):
        # The mesh must be reached through the pybind11 R3DWidget rather than
        # QMdiSubWindow.widget(), which returns a bare QWidget.
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        current = self.mgr.currentR3DWidget()
        self.assertIsNotNone(current)
        self.assertIsNotNone(current.mesh)
        self.assertEqual(current.mesh.ncell, 3)

    def test_panel_shows_active_mesh(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        feature = _mesh_info.MeshInfo(mgr=self.mgr, menu=self.menu)
        feature.populate_menu()
        self.assertIn(feature._action, self.menu.actions())
        feature._action.setChecked(True)
        sections = _tree_sections(feature._panel._tree)
        self.assertEqual(sections["Counts"]["cell"], "3")
        self.assertEqual(sections["Cell types"]["triangle"], "2")

    def test_panel_without_mesh(self):
        self.mgr.add3DWidget()  # fresh viewer becomes current, no mesh
        feature = _mesh_info.MeshInfo(mgr=self.mgr, menu=self.menu)
        feature.populate_menu()
        feature._action.setChecked(True)
        root = feature._panel._tree.topLevelItem(0)
        self.assertIn("No mesh", root.text(0))
        self.assertEqual(root.childCount(), 0)

    def test_panel_updates_on_menu_load(self):
        # Loading from a menu creates and activates the viewer before
        # updateMesh runs, so the refresh is deferred to the event loop.
        feature = _mesh_info.MeshInfo(mgr=self.mgr, menu=self.menu)
        feature.populate_menu()
        feature._action.setChecked(True)
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_single_triangle())
        QApplication.processEvents()  # allow the deferred refresh to run
        sections = _tree_sections(feature._panel._tree)
        self.assertEqual(sections["Counts"]["cell"], "1")
        self.assertEqual(sections["Cell types"]["triangle"], "1")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
