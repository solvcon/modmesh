import json
from typing import Any

from ._gui_common import PilotFeature

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QTableWidgetItem,
    QTableWidget,
    QWidget,
    QHeaderView,
)
from modmesh.profiling import ProfilingResultPrinter

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

        self._table_widget = QTableWidget()
        self._table_widget.setWindowTitle(f"Profiling Result: {file_name}")
        self._table_widget.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        style: str = """
            QTableView {
                background: white;
                color: black;
            }
        """

        self._table_widget.setStyleSheet(style)

        printer = ProfilingResultPrinter(result)

        printer.add_column("total_time", lambda r: r.total_time)

        column_data = printer.column_data
        column_names = [column.column_name for column in printer.column_data]
        column_count = len(column_names)

        self._table_widget.setColumnCount(len(column_names))
        self._table_widget.setHorizontalHeaderLabels(column_names)
        self._table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        if len(column_names) != 0:
            first_column = printer.column_data[0]
            row_count = len(first_column.column_value)
            self._table_widget.setRowCount(row_count)

            for column, c_index in zip(column_data, range(column_count)):
                column_value: list[str | float] = column.column_value
                for value, r_index in zip(column_value, range(row_count)):
                    item = QTableWidgetItem(str(value))
                    self._table_widget.setItem(r_index, c_index, item)

        self._table.setWidget(self._table_widget)
        # self._table.showMaximized()
        self._table.show()
