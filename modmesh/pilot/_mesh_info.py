# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Dock panel showing StaticMesh geometry counts in a foldable tree."""

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QDockWidget,
                               QTreeWidget, QTreeWidgetItem, QFrame)

from .. import core
from . import _gui_common

__all__ = [  # noqa: F822
    'MeshInfo',
]


class MeshInfoPanel(QWidget):
    """Widget that presents the mesh information tree inside the dock."""

    # Map cell type numbers to human-readable names.
    CELL_TYPE_NAME = {
        core.StaticMesh.POINT: "point",
        core.StaticMesh.LINE: "line",
        core.StaticMesh.QUADRILATERAL: "quadrilateral",
        core.StaticMesh.TRIANGLE: "triangle",
        core.StaticMesh.HEXAHEDRON: "hexahedron",
        core.StaticMesh.TETRAHEDRON: "tetrahedron",
        core.StaticMesh.PRISM: "prism",
        core.StaticMesh.PYRAMID: "pyramid",
    }

    def __init__(self, mh=None, parent=None):
        super().__init__(parent)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        # Drop the tree frame so its scroll bar sits flush in the panel.
        self._tree.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)
        self.setLayout(layout)
        self.set_mesh(mh)

    @classmethod
    def make_mesh_info(cls, mh):
        """Build the mesh information as ``(section, rows)`` groups.

        Each group pairs a heading with its ``[property, value]`` string
        rows, so the panel renders one foldable tree node per group.
        """
        sections = [
            ("Counts", [
                ["dim", str(mh.ndim)],
                ["node", str(mh.nnode)],
                ["face", str(mh.nface)],
                ["cell", str(mh.ncell)],
                ["edge", str(mh.nedge)],
                ["bound", str(mh.nbound)],
                ["bcs", str(mh.nbcs)],
            ]),
            ("Ghost", [
                ["node", str(mh.ngstnode)],
                ["face", str(mh.ngstface)],
                ["cell", str(mh.ngstcell)],
            ]),
        ]
        # Ghost entities are stored first; measure only the body entities.
        crd = mh.ndcrd.ndarray[mh.ndcrd.nghost:]
        if crd.size:
            lower = crd.min(axis=0)
            upper = crd.max(axis=0)
            bbox = [[axis, f"[{lower[it]:.4g}, {upper[it]:.4g}]"]
                    for it, axis in zip(range(mh.ndim), "xyz")]
            sections.append(("Bounding box", bbox))
        # Tally the cell types over the body (non-ghost) cells.
        tpn = mh.cltpn.ndarray[mh.cltpn.nghost:]
        cells = []
        for tnum, name in sorted(cls.CELL_TYPE_NAME.items()):
            count = int(np.count_nonzero(tpn == tnum))
            if count:
                cells.append([name, str(count)])
        if cells:
            sections.append(("Cell types", cells))
        return sections

    def set_mesh(self, mh):
        """Rebuild the tree from ``mh``, or show "No mesh loaded" when None."""
        self._tree.clear()
        if mh is None:
            QTreeWidgetItem(self._tree, ["No mesh loaded"])
            return
        root = QTreeWidgetItem(self._tree, [f"StaticMesh ({mh.ndim}D)"])
        for section, rows in self.make_mesh_info(mh):
            group = QTreeWidgetItem(root, [section])
            for prop, value in rows:
                QTreeWidgetItem(group, [f"{prop}: {value}"])
            group.setExpanded(True)
        root.setExpanded(True)
        # Widen the column so long entries are not clipped.
        self._tree.resizeColumnToContents(0)


class MeshInfo(_gui_common.PilotFeature):
    """Mesh information panel, toggled from the View "Panels" submenu.

    The caller supplies the ``menu`` group the toggle item is added to. When
    on, the panel shows the active sub-window's mesh and follows the active
    sub-window; sub-windows without a mesh show "No mesh loaded".
    """

    def __init__(self, *args, **kw):
        self._menu = kw.pop('menu')
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None

    def populate_menu(self):
        self._action = QAction("Mesh", self._mainWindow)
        self._action.setStatusTip("Toggle the mesh information panel")
        self._action.setCheckable(True)
        self._action.toggled.connect(self._on_toggled)
        self._menu.addAction(self._action)

    def _on_toggled(self, checked):
        """Show or hide the panel."""
        if checked:
            self._ensure_panel()
            self._refresh()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        """Build the dock lazily and follow sub-window activation."""
        if self._panel is not None:
            return
        self._panel = MeshInfoPanel()
        self._dock = QDockWidget("mesh")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           self._dock)
        # Keep the menu check in sync when the dock is closed by its button.
        self._dock.visibilityChanged.connect(self._action.setChecked)
        mdi = self._mdi_area()
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_subwindow_activated(self, _subwin):
        """Refresh the panel when the active sub-window changes.

        The refresh is deferred to the next event-loop pass because loading
        a mesh from a menu activates the new sub-window before ``updateMesh``
        populates it.
        """
        if self._dock is not None and self._action.isChecked():
            QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        """Show the active sub-window's mesh."""
        self._panel.set_mesh(self._active_mesh())

    def _mdi_area(self):
        return self._mainWindow.centralWidget()

    def _active_mesh(self):
        """Return the active 3D viewer's mesh, or ``None``.

        ``RManager.currentR3DWidget`` is used so the pybind11 widget that
        exposes ``mesh`` is reached; ``QMdiSubWindow.widget()`` would return
        a bare ``QWidget``.
        """
        widget = self._mgr.currentR3DWidget()
        return None if widget is None else widget.mesh

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
