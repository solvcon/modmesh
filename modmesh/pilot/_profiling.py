import json
from typing import Any

from ._gui_common import PilotFeature

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTreeView,
    QWidget
)
from PySide6.QtGui import (
    QStandardItem,
    QStandardItemModel
)

__all__ = ["Profiling"]


class Profiling(PilotFeature):
    """
    Create profiling windows.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setWindowTitle("Open profiling file")

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.profilingMenu,
            text="Open profiling result",
            tip="Open JSON file of profiling result",
            func=self.open_profiling_result,
        )

    def open_profiling_result(self):
        self._diag.open(self, QtCore.SLOT("on_finished()"))

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)

        result = None
        with open(filenames[0], "r") as f:
            result = json.loads(f.read())

        self._add_result_window(filenames[0], result)

    def _add_result_window(self, file_name: str, result: list[dict[str, Any]]):
        self._table = self._mgr.addSubWindow(QWidget())
        self._tree_view = QTreeView(self._table)

        self._model = QStandardItemModel(self._tree_view)
        self._model.setHorizontalHeaderLabels(["Total Time", "Symbol Name"])

        def _recursive_add_item(
                parent: QStandardItem,
                data: dict[str, Any],
                total_time: float
        ):
            percent = data["total_time"] * 100 / total_time

            first_item = QStandardItem(f"{data['total_time']} ({percent:2f}%)")
            second_item = QStandardItem(data["name"])

            parent.appendRow([first_item, second_item])

            sorted_child_data = sorted(
                data["children"],
                key=lambda d: d["total_time"],
                reverse=True
            )

            for child_data in sorted_child_data:
                _recursive_add_item(first_item, child_data, data["total_time"])

        for data in result:
            first_item = QStandardItem(f"{data['total_time']} (100%)")
            second_item = QStandardItem(data["name"])

            self._model.appendRow([first_item, second_item])

            for child_data in data["children"]:
                _recursive_add_item(first_item, child_data, data['total_time'])

        self._tree_view.setStyleSheet(
            """
            QTreeView {
                background-color: #2e2e2e;
                color: #dcdcdc;
                border: 1px solid #555555;
            }
            QTreeView::item {
                border: 1px solid #3c3c3c;
                border-right: none;
                border-bottom: none;
            }
            QTreeView::item:selected {
                background-color: #4f4f4f;
                color: white;
            }
            """
        )

        self._tree_view.setModel(self._model)
        self._tree_view.setColumnWidth(0, 400)
        self._tree_view.setColumnWidth(1, 200)
        self._tree_view.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self._table.setWidget(self._tree_view)
        self._table.showMaximized()
        self._table.show()
